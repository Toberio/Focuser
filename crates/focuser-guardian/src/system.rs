//! What the guardian asks of the OS: who is logged in, is Focuser running
//! for them, and starting it again. Linux only — Windows and macOS have no
//! guardian yet, and the binary refuses to start there.

use std::path::PathBuf;

/// A user with an active login session.
#[derive(Debug)]
pub struct User {
    pub uid: u32,
    pub name: String,
    pub db_path: PathBuf,
    /// Their session has a display, so a window would actually open.
    pub session_ready: bool,
}

pub use imp::{active_users, ensure_supported, revive, ui_running};

#[cfg(target_os = "linux")]
mod imp {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use anyhow::{Result, bail};
    use tracing::debug;

    use super::User;

    const UNIT: &str = "focuser.service";

    /// Regular accounts start here on every mainstream distribution; below
    /// it are system users, who never run Focuser.
    const FIRST_REGULAR_UID: u32 = 1000;

    pub fn ensure_supported() -> Result<()> {
        let uid = status_uid(Path::new("/proc/self/status")).unwrap_or(u32::MAX);
        if uid != 0 {
            bail!("focuser-guardian must run as root (it is started by focuser-guardian.service)");
        }
        Ok(())
    }

    pub fn active_users() -> Vec<User> {
        let Ok(entries) = std::fs::read_dir("/run/user") else {
            return Vec::new();
        };
        entries
            .filter_map(|e| e.ok()?.file_name().to_str()?.parse::<u32>().ok())
            .filter(|uid| *uid >= FIRST_REGULAR_UID)
            .filter(|uid| session_active(*uid))
            .filter_map(user)
            .collect()
    }

    /// `active` excludes `closing` (logging out) and `lingering` (no
    /// session at all), the two states in which a revive would be wrong.
    fn session_active(uid: u32) -> bool {
        output(
            "loginctl",
            &["show-user", &uid.to_string(), "--property=State", "--value"],
        )
        .is_some_and(|s| s.trim() == "active")
    }

    fn user(uid: u32) -> Option<User> {
        // getent rather than /etc/passwd, so network accounts resolve too.
        let entry = output("getent", &["passwd", &uid.to_string()])?;
        let fields: Vec<&str> = entry.trim().split(':').collect();
        let (name, home) = (fields.first()?.to_string(), PathBuf::from(fields.get(5)?));

        let env = user_systemctl(&name, &["show-environment"]).unwrap_or_default();
        let var = |key: &str| {
            env.lines()
                .find_map(|l| l.strip_prefix(key)?.strip_prefix('='))
                .map(str::to_string)
        };
        let session_ready = var("DISPLAY").is_some() || var("WAYLAND_DISPLAY").is_some();
        // Where `directories::ProjectDirs` puts it for the app itself.
        let data_home = var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local/share"));

        Some(User {
            uid,
            name,
            db_path: data_home.join("focuser").join("focuser.db"),
            session_ready,
        })
    }

    /// The installed app. Only a process running this file counts.
    const UI_EXE: &str = "/usr/bin/focuser-ui";

    /// Is the installed focuser-ui running as this user?
    ///
    /// Judged by the executable, not the name: a process name is whatever its
    /// owner calls it, so `cp /bin/sleep ~/focuser-ui` (or a dev build that
    /// blocks nothing) would otherwise stand in for the real app.
    pub fn ui_running(uid: u32) -> bool {
        let Ok(entries) = std::fs::read_dir("/proc") else {
            // Unknown is not "stopped": reviving blind would stack copies.
            return true;
        };
        entries.filter_map(|e| e.ok()).any(|e| {
            let Ok(exe) = std::fs::read_link(e.path().join("exe")) else {
                return false;
            };
            if !is_installed_ui(&exe.to_string_lossy()) {
                return false;
            }
            let Ok(status) = std::fs::read_to_string(e.path().join("status")) else {
                return false;
            };
            let field = |key: &str| {
                status
                    .lines()
                    .find_map(|l| l.strip_prefix(key))
                    .map(str::trim)
            };
            !field("State:").is_some_and(|s| s.starts_with('Z'))
                && parse_uid(field("Uid:")) == Some(uid)
        })
    }

    /// An upgrade renames a new binary over the old one, and a process still
    /// running the old one then reads as `... (deleted)`. That is the real
    /// app too; not counting it would revive a unit that is already active,
    /// every cooldown, until the next restart.
    fn is_installed_ui(exe: &str) -> bool {
        exe == UI_EXE || exe.strip_suffix(" (deleted)") == Some(UI_EXE)
    }

    fn status_uid(path: &Path) -> Option<u32> {
        let status = std::fs::read_to_string(path).ok()?;
        parse_uid(status.lines().find_map(|l| l.strip_prefix("Uid:")))
    }

    /// The real uid, the first of the four numbers on the `Uid:` line.
    fn parse_uid(field: Option<&str>) -> Option<u32> {
        field?.split_whitespace().next()?.parse().ok()
    }

    /// Put the user's unit back the way the package shipped it and start it.
    ///
    /// `revert` is what makes this more than a start: it removes a mask, a
    /// `systemctl --user edit` drop-in and a full override copy in
    /// `~/.config/systemd/user` alike, any of which would otherwise keep the
    /// unit from running the real app. `reset-failed` clears a restart limit
    /// used up by repeated kills.
    pub fn revive(name: &str) -> Result<()> {
        for args in [
            &["revert", UNIT][..],
            &["daemon-reload"],
            &["reset-failed", UNIT],
        ] {
            if user_systemctl(name, args).is_none() {
                debug!(user = name, ?args, "systemctl step failed, carrying on");
            }
        }
        // --no-block: the guardian must not hang on a slow start.
        if user_systemctl(name, &["start", "--no-block", UNIT]).is_none() {
            bail!("systemctl --user start {UNIT} failed");
        }
        Ok(())
    }

    /// Run `systemctl --user` against another user's manager, as root.
    fn user_systemctl(name: &str, args: &[&str]) -> Option<String> {
        let machine = format!("--machine={name}@.host");
        let mut full = vec!["--user", machine.as_str()];
        full.extend_from_slice(args);
        output("systemctl", &full)
    }

    fn output(program: &str, args: &[&str]) -> Option<String> {
        let out = Command::new(program).args(args).output().ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    }

    #[cfg(test)]
    mod tests {
        use super::{is_installed_ui, parse_uid};

        #[test]
        fn only_the_installed_binary_counts() {
            assert!(is_installed_ui("/usr/bin/focuser-ui"));
            assert!(is_installed_ui("/usr/bin/focuser-ui (deleted)"));
            assert!(!is_installed_ui("/home/u/focuser-ui"));
            assert!(!is_installed_ui(
                "/home/u/code/Focuser/target/release/focuser-ui"
            ));
            assert!(!is_installed_ui("/usr/bin/focuser-ui.new"));
            assert!(!is_installed_ui("/usr/bin/sleep"));
        }

        #[test]
        fn reads_the_real_uid() {
            assert_eq!(parse_uid(Some("\t1000\t1000\t1000\t1000")), Some(1000));
            assert_eq!(parse_uid(Some("0 0 0 0")), Some(0));
            assert_eq!(parse_uid(Some("")), None);
            assert_eq!(parse_uid(None), None);
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use anyhow::{Result, bail};

    use super::User;

    pub fn ensure_supported() -> Result<()> {
        bail!("focuser-guardian is only implemented for Linux")
    }

    pub fn active_users() -> Vec<User> {
        Vec::new()
    }

    pub fn ui_running(_uid: u32) -> bool {
        true
    }

    pub fn revive(_name: &str) -> Result<()> {
        Ok(())
    }
}
