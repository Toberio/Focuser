//! Launch at login. Always on: a blocker that only runs when you remember to
//! open it is not much of a blocker, so there is no toggle.
//!
//! Every login launcher passes [`FLAG`], which starts Focuser in the tray with
//! no window. Opening it from the Start menu or a launcher shows the window.
//!
//! What starts it depends on the platform:
//! - **Windows**: the NSIS installer creates a `/rl highest` scheduled task, so
//!   Focuser starts elevated and can write the hosts file without a UAC prompt.
//!   The plugin's `HKCU\...\Run` entry is only for portable and dev builds,
//!   which never had the task.
//! - **macOS**: the plugin's LaunchAgent.
//! - **Linux**: the packaged systemd user unit, which also respawns Focuser if
//!   it is killed. The plugin's XDG autostart entry is only a fallback for
//!   installs without the unit; with both, login launched Focuser twice and
//!   the second launch made the single-instance handler open the window.

use tauri::AppHandle;
use tauri_plugin_autostart::ManagerExt;
use tracing::warn;

/// Passed by every login launcher. Means "start in the tray".
pub const FLAG: &str = "--autostart";

/// Whether this process was started by a login launcher rather than the user.
pub fn launched_at_login<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    args.into_iter().any(|a| a.as_ref() == FLAG)
}

/// Make sure Focuser starts at login, undoing an old "off" from when this was
/// a setting.
pub fn ensure_enabled(app: &AppHandle) {
    let plugin = app.autolaunch();
    let result = if imp::has_own_launcher() {
        plugin.disable()
    } else {
        plugin.enable()
    };
    if let Err(e) = result {
        warn!("autostart plugin refused: {e}");
    }
    imp::enable_task();
}

#[cfg(windows)]
mod imp {
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Output};
    use tracing::{info, warn};

    const TASK: &str = "Focuser";
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    fn schtasks(args: &[&str]) -> Option<Output> {
        Command::new("schtasks")
            .args(args)
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .map_err(|e| warn!("schtasks failed to run: {e}"))
            .ok()
    }

    fn query() -> Option<String> {
        let out = schtasks(&["/query", "/tn", TASK, "/fo", "list"])?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).to_ascii_lowercase())
    }

    /// With the task installed the Run entry has to go: it starts Focuser
    /// unelevated, and if it wins the race the elevated task launch is the one
    /// the single-instance plugin drops.
    pub fn has_own_launcher() -> bool {
        query().is_some()
    }

    /// Re-enable the logon task if the old toggle disabled it. Changing a
    /// `/rl highest` task needs admin, so this can fail when opened from the
    /// Start menu; it is retried on every start.
    pub fn enable_task() {
        // "Status" is localised, but a disabled task always says Disabled.
        let Some(status) = query() else { return };
        if !status.contains("disabled") {
            return;
        }
        match schtasks(&["/change", "/tn", TASK, "/enable"]) {
            Some(out) if out.status.success() => info!("logon task re-enabled"),
            Some(out) => warn!(
                "could not enable the logon task: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ),
            None => {}
        }
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use std::path::Path;

    /// Where the .deb puts the unit. Tied to the packaging in tauri.conf.json.
    const SYSTEMD_UNIT: &str = "/usr/lib/systemd/user/focuser.service";

    pub fn has_own_launcher() -> bool {
        Path::new(SYSTEMD_UNIT).exists()
    }

    pub fn enable_task() {}
}

#[cfg(target_os = "macos")]
mod imp {
    pub fn has_own_launcher() -> bool {
        false
    }

    pub fn enable_task() {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_login_flag_is_recognised() {
        assert!(launched_at_login(["focuser-ui", "--autostart"]));
    }

    #[test]
    fn a_normal_launch_is_not_a_login_launch() {
        assert!(!launched_at_login(["focuser-ui"]));
        assert!(!launched_at_login(["focuser-ui", "--export-bindings"]));
    }
}
