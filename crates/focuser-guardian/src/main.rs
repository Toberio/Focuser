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
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use focuser_common::types::BlockList;
use focuser_core::Database;
use tracing::{info, warn};

/// How often every user is looked at.
const TICK: Duration = Duration::from_secs(2);

/// After a revive, how long to give the app to show up before trying again.
/// A start that worked but has not produced a process yet must not be
/// stacked with another.
const COOLDOWN: Duration = Duration::from_secs(10);

/// How long a user's name and database path are trusted before being looked
/// up again. Each lookup starts `getent` and `systemctl`; doing that every
/// tick cost a few percent of a core for nothing.
const USER_REFRESH: Duration = Duration::from_secs(300);

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

/// `protected` is `None` when the database could not be read this tick.
/// The other two are asked only once a revive is otherwise due: both scan
/// or spawn, and with no lock on (nearly always) neither is needed.
fn decide(
    protected: Option<bool>,
    ui_running: impl FnOnce() -> bool,
    session_ready: impl FnOnce() -> bool,
    state: &mut UserState,
    now: Instant,
) -> Action {
    if let Some(p) = protected {
        state.protected = p;
    }
    if !state.protected || state.cooldown_until.is_some_and(|until| now < until) {
        return Action::Nothing;
    }
    if ui_running() || !session_ready() {
        return Action::Nothing;
    }
    state.cooldown_until = Some(now + COOLDOWN);
    Action::Revive
}

/// A user's block lists, re-read only when the database file changes.
///
/// The lists are kept rather than the answer, because whether one is
/// protected depends on the time: schedules and timers start and end
/// without the file changing.
#[derive(Default)]
struct DbCache {
    /// Modification time and length when `lists` was read.
    stamp: Option<(SystemTime, u64)>,
    lists: Vec<BlockList>,
}

impl DbCache {
    /// Does any list in this database forbid stopping Focuser right now?
    fn service_protected(&mut self, db_path: &Path) -> Option<bool> {
        let meta = match std::fs::metadata(db_path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                *self = Self::default();
                return Some(false);
            }
            Err(e) => {
                warn!(path = %db_path.display(), "could not stat database: {e}");
                return None;
            }
        };
        let stamp = meta.modified().ok().map(|t| (t, meta.len()));
        if stamp.is_none() || stamp != self.stamp {
            match Database::open_read_only(db_path).and_then(|db| db.list_block_lists()) {
                Ok(lists) => {
                    self.lists = lists;
                    self.stamp = stamp;
                }
                Err(e) => {
                    // Read again next tick rather than trusting a stale copy.
                    self.stamp = None;
                    warn!(path = %db_path.display(), "could not read database: {e}");
                    return None;
                }
            }
        }
        Some(self.lists.iter().any(|l| l.has_service_protection()))
    }
}

/// Everything kept about one logged-in user.
struct Tracked {
    user: system::User,
    looked_up: Instant,
    db: DbCache,
    state: UserState,
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

    let mut tracked: HashMap<u32, Tracked> = HashMap::new();
    loop {
        let uids = system::active_uids();
        tracked.retain(|uid, _| uids.contains(uid));

        for uid in uids {
            let stale = tracked
                .get(&uid)
                .is_none_or(|t| t.looked_up.elapsed() >= USER_REFRESH);
            if stale {
                let Some(user) = system::user(uid) else {
                    continue;
                };
                let t = tracked.entry(uid).or_insert_with(|| Tracked {
                    user: user.clone(),
                    looked_up: Instant::now(),
                    db: DbCache::default(),
                    state: UserState::default(),
                });
                if t.user.db_path != user.db_path {
                    t.db = DbCache::default();
                }
                t.user = user;
                t.looked_up = Instant::now();
            }
            let Some(t) = tracked.get_mut(&uid) else {
                continue;
            };

            let protected = t.db.service_protected(&t.user.db_path);
            let action = decide(
                protected,
                || system::ui_running(uid),
                || system::session_ready(&t.user.name),
                &mut t.state,
                Instant::now(),
            );
            if action == Action::Revive {
                info!(user = %t.user.name, "Focuser is not running under a lock; starting it");
                if let Err(e) = system::revive(&t.user.name) {
                    warn!(user = %t.user.name, "could not start Focuser: {e}");
                }
            }
        }

        std::thread::sleep(TICK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        protected: Option<bool>,
        running: bool,
        ready: bool,
        s: &mut UserState,
        now: Instant,
    ) -> Action {
        decide(protected, || running, || ready, s, now)
    }

    #[test]
    fn revives_only_when_locked_stopped_and_ready() {
        let now = Instant::now();
        assert_eq!(
            run(Some(false), false, true, &mut UserState::default(), now),
            Action::Nothing
        );
        assert_eq!(
            run(Some(true), true, true, &mut UserState::default(), now),
            Action::Nothing
        );
        assert_eq!(
            run(Some(true), false, false, &mut UserState::default(), now),
            Action::Nothing
        );
        assert_eq!(
            run(Some(true), false, true, &mut UserState::default(), now),
            Action::Revive
        );
    }

    #[test]
    fn with_no_lock_nothing_is_scanned_or_spawned() {
        let mut s = UserState::default();
        let action = decide(
            Some(false),
            || panic!("scanned /proc with no lock on"),
            || panic!("asked the session with no lock on"),
            &mut s,
            Instant::now(),
        );
        assert_eq!(action, Action::Nothing);
    }

    #[test]
    fn waits_out_the_cooldown_before_trying_again() {
        let now = Instant::now();
        let mut s = UserState::default();
        assert_eq!(run(Some(true), false, true, &mut s, now), Action::Revive);
        assert_eq!(
            run(Some(true), false, true, &mut s, now + TICK),
            Action::Nothing
        );
        assert_eq!(
            run(Some(true), false, true, &mut s, now + COOLDOWN),
            Action::Revive
        );
    }

    #[test]
    fn an_unreadable_database_keeps_the_last_answer() {
        let now = Instant::now();
        let mut s = UserState::default();
        run(Some(true), true, true, &mut s, now);
        assert_eq!(run(None, false, true, &mut s, now), Action::Revive);

        let mut s = UserState::default();
        assert_eq!(run(None, false, true, &mut s, now), Action::Nothing);
    }

    #[test]
    fn a_lock_ending_stops_the_revives() {
        let now = Instant::now();
        let mut s = UserState::default();
        run(Some(true), true, true, &mut s, now);
        assert_eq!(run(Some(false), false, true, &mut s, now), Action::Nothing);
    }

    #[test]
    fn the_database_is_read_again_only_after_it_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("focuser.db");
        let mut cache = DbCache::default();
        assert_eq!(cache.service_protected(&path), Some(false));

        let db = Database::open(&path).unwrap();
        db.create_block_list(&BlockList::new("Work")).unwrap();
        assert_eq!(cache.service_protected(&path), Some(false));
        assert_eq!(cache.lists.len(), 1);

        // Swap the cached copy for a marker: an unchanged file must not be re-read.
        cache.lists.clear();
        assert_eq!(cache.service_protected(&path), Some(false));
        assert!(cache.lists.is_empty());

        db.create_block_list(&BlockList::new("Play")).unwrap();
        cache.service_protected(&path);
        assert_eq!(cache.lists.len(), 2);
    }
}
