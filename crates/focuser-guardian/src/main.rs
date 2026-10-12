//! focuser-guardian — keeps Focuser running while a lock says it must.
//!
//! Focuser runs as the user, under a systemd *user* unit that restarts it
//! when it is killed. Anything running as that same user can still get past
//! that: `systemctl --user stop` (systemd never restarts its own stop),
//! `systemctl --user mask`, or five kills inside a minute, which uses up the
//! unit's restart limit. A lock with "prevent service stop" promised more
//! than that.
//!
//! This binary runs as root under a *system* unit, so the user cannot stop
//! it without sudo. Every few seconds it looks at each logged-in user's
//! database, and when one of their lists has service protection on and
//! focuser-ui is not running, it reverts, resets and starts their unit
//! again. With no lock on, it does nothing at all, so stopping Focuser
//! normally stays possible.
//!
//! It only ever reads the database ([`Database::open_read_only`]), and only
//! acts once the user's session has a display, so it cannot start the window
//! during login or logout before or after there is a screen to put it on.

mod system;

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use focuser_core::Database;
use tracing::{info, warn};

/// How often every user is looked at.
const TICK: Duration = Duration::from_secs(2);

/// After a revive, how long to give the app to show up before trying again.
/// A start that worked but has not produced a process yet must not be
/// stacked with another.
const COOLDOWN: Duration = Duration::from_secs(10);

/// What one tick saw for one user.
#[derive(Debug, Clone, Copy)]
struct Observation {
    /// `None` when the database could not be read this tick (busy, mid-write).
    service_protected: Option<bool>,
    ui_running: bool,
    session_ready: bool,
}

/// What the guardian remembers about one user between ticks.
#[derive(Debug, Default)]
struct UserState {
    /// The last answer the database gave, used when it cannot be read.
    /// Failing open on a busy database would make a write lock a way out.
    protected: bool,
    cooldown_until: Option<Instant>,
}

#[derive(Debug, PartialEq, Eq)]
enum Action {
    Nothing,
    Revive,
}

fn decide(obs: Observation, state: &mut UserState, now: Instant) -> Action {
    if let Some(p) = obs.service_protected {
        state.protected = p;
    }
    if !state.protected || obs.ui_running || !obs.session_ready {
        return Action::Nothing;
    }
    if state.cooldown_until.is_some_and(|until| now < until) {
        return Action::Nothing;
    }
    state.cooldown_until = Some(now + COOLDOWN);
    Action::Revive
}

/// Does any list in this database forbid stopping Focuser right now?
fn service_protected(db_path: &Path) -> Option<bool> {
    if !db_path.exists() {
        return Some(false);
    }
    let lists = Database::open_read_only(db_path)
        .and_then(|db| db.list_block_lists())
        .map_err(|e| warn!(path = %db_path.display(), "could not read database: {e}"))
        .ok()?;
    Some(lists.iter().any(|l| l.has_service_protection()))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "focuser_guardian=info".into()),
        )
        // journald stamps every line already.
        .without_time()
        .init();

    system::ensure_supported()?;
    info!("focuser-guardian started");

    let mut states: HashMap<u32, UserState> = HashMap::new();
    loop {
        let users = system::active_users();
        states.retain(|uid, _| users.iter().any(|u| u.uid == *uid));

        for user in &users {
            let obs = Observation {
                service_protected: service_protected(&user.db_path),
                ui_running: system::ui_running(user.uid),
                session_ready: user.session_ready,
            };
            let state = states.entry(user.uid).or_default();
            if decide(obs, state, Instant::now()) == Action::Revive {
                info!(user = %user.name, "Focuser is not running under a lock; starting it");
                if let Err(e) = system::revive(&user.name) {
                    warn!(user = %user.name, "could not start Focuser: {e}");
                }
            }
        }

        std::thread::sleep(TICK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(protected: Option<bool>, running: bool, ready: bool) -> Observation {
        Observation {
            service_protected: protected,
            ui_running: running,
            session_ready: ready,
        }
    }

    #[test]
    fn revives_only_when_locked_stopped_and_ready() {
        let now = Instant::now();
        let mut s = UserState::default();
        assert_eq!(
            decide(obs(Some(false), false, true), &mut s, now),
            Action::Nothing
        );
        let mut s = UserState::default();
        assert_eq!(
            decide(obs(Some(true), true, true), &mut s, now),
            Action::Nothing
        );
        let mut s = UserState::default();
        assert_eq!(
            decide(obs(Some(true), false, false), &mut s, now),
            Action::Nothing
        );
        let mut s = UserState::default();
        assert_eq!(
            decide(obs(Some(true), false, true), &mut s, now),
            Action::Revive
        );
    }

    #[test]
    fn waits_out_the_cooldown_before_trying_again() {
        let now = Instant::now();
        let mut s = UserState::default();
        let stopped = obs(Some(true), false, true);
        assert_eq!(decide(stopped, &mut s, now), Action::Revive);
        assert_eq!(decide(stopped, &mut s, now + TICK), Action::Nothing);
        assert_eq!(decide(stopped, &mut s, now + COOLDOWN), Action::Revive);
    }

    #[test]
    fn an_unreadable_database_keeps_the_last_answer() {
        let now = Instant::now();
        let mut s = UserState::default();
        decide(obs(Some(true), true, true), &mut s, now);
        assert_eq!(decide(obs(None, false, true), &mut s, now), Action::Revive);

        let mut s = UserState::default();
        assert_eq!(decide(obs(None, false, true), &mut s, now), Action::Nothing);
    }

    #[test]
    fn a_lock_ending_stops_the_revives() {
        let now = Instant::now();
        let mut s = UserState::default();
        decide(obs(Some(true), true, true), &mut s, now);
        assert_eq!(
            decide(obs(Some(false), false, true), &mut s, now),
            Action::Nothing
        );
    }
}
