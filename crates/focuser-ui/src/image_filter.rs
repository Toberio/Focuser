//! The explicit-image filter's classifier, run by the app for every browser.
//!
//! The browser extension finds the images; this judges them, on the GPU, and
//! answers over the local API. Nothing leaves the machine except the one-off
//! model download from Hugging Face the first time a list turns the filter on.
//!
//! Lifecycle: the extension polls `/api/rules` every couple of seconds, and
//! that poll calls [`sync`]. While any active list has the filter on, the
//! models are downloaded if missing and loaded; once none does, they are
//! dropped again, which frees the GPU memory they hold.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use focuser_common::types::ImageFilter;
use focuser_vision::{Classifier, models};
use tracing::{error, info};

pub use focuser_common::types::ImageFilterStatus as Status;

enum State {
    Off,
    Starting(Status),
    Ready(Classifier),
    Failed(String),
}

struct Service {
    dir: PathBuf,
    state: Mutex<State>,
}

static SERVICE: OnceLock<Service> = OnceLock::new();

/// Where the models live. Called once at startup.
pub fn init(models_dir: PathBuf) {
    let _ = SERVICE.set(Service {
        dir: models_dir,
        state: Mutex::new(State::Off),
    });
}

fn service() -> Option<&'static Service> {
    SERVICE.get()
}

/// Match the classifier to the filter level the rules ask for.
pub fn sync(level: ImageFilter) {
    let Some(svc) = service() else { return };
    let Ok(mut state) = svc.state.lock() else {
        return;
    };
    match (&*state, level) {
        (State::Off, l) if l != ImageFilter::Off => {
            *state = State::Starting(Status::Loading);
            drop(state);
            start(svc);
        }
        (State::Ready(_) | State::Failed(_), ImageFilter::Off) => {
            info!("image filter off; releasing its models");
            *state = State::Off;
        }
        _ => {}
    }
}

fn set(svc: &Service, next: State) {
    if let Ok(mut state) = svc.state.lock() {
        // Turned off while starting: stay off rather than load anyway.
        if matches!(*state, State::Off) && !matches!(next, State::Off) {
            return;
        }
        *state = next;
    }
}

fn start(svc: &'static Service) {
    let spawned = std::thread::Builder::new()
        .name("image-filter-load".into())
        .spawn(move || {
            let mut last = 0u64;
            let downloaded = models::ensure(&svc.dir, &mut |done, total| {
                // A status update per megabyte is plenty.
                if done == total || done - last >= 1 << 20 {
                    last = done;
                    set(
                        svc,
                        State::Starting(Status::Downloading {
                            done_mb: (done >> 20) as u32,
                            total_mb: (total >> 20) as u32,
                        }),
                    );
                }
            });
            if let Err(e) = downloaded {
                error!(error = %e, "image filter model download failed");
                set(svc, State::Failed(e.to_string()));
                return;
            }
            set(svc, State::Starting(Status::Loading));
            match Classifier::load(&svc.dir) {
                Ok(classifier) => {
                    set(svc, State::Ready(classifier));
                    // The user's own filter, from labels saved last time.
                    crate::image_feedback::retrain();
                }
                Err(e) => {
                    error!(error = %e, "image filter models failed to load");
                    set(svc, State::Failed(e.to_string()));
                }
            }
        });
    if let Err(e) = spawned {
        set(svc, State::Failed(e.to_string()));
    }
}

pub fn status() -> Status {
    let Some(svc) = service() else {
        return Status::Off;
    };
    let Ok(state) = svc.state.lock() else {
        return Status::Off;
    };
    status_of(&state)
}

/// Judge one image, or an animation's first frame: scores and SigLIP's
/// embedding. `Err` carries the status when the models are not ready.
pub fn judge(bytes: &[u8]) -> Result<focuser_vision::Judged, Status> {
    ready()?.judge(bytes).map_err(|e| Status::Failed {
        error: e.to_string(),
    })
}

/// Judge every frame worth judging, one for a still image and several spread
/// through an animation: scores and SigLIP's embedding of each. `Err` carries
/// the status when the models are not ready.
pub fn judge_frames(bytes: &[u8]) -> Result<Vec<focuser_vision::Judged>, Status> {
    ready()?.judge_frames(bytes).map_err(|e| Status::Failed {
        error: e.to_string(),
    })
}

fn ready() -> Result<Classifier, Status> {
    let svc = service().ok_or(Status::Off)?;
    let state = svc.state.lock().map_err(|_| Status::Off)?;
    match &*state {
        // Cloned out so the lock is not held through inference.
        State::Ready(c) => Ok(c.clone()),
        _ => Err(status_of(&state)),
    }
}

fn status_of(state: &State) -> Status {
    match state {
        State::Off => Status::Off,
        State::Starting(s) => s.clone(),
        State::Ready(_) => Status::Ready,
        State::Failed(e) => Status::Failed { error: e.clone() },
    }
}
