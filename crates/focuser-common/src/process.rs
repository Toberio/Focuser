//! Listing and terminating running processes.
//!
//! App blocking needs two primitives on every OS: enumerate what is running,
//! and stop one of them. The three callers in the GUI's blocking loop differ
//! only in *which* processes they pick, so the OS-specific part lives here and
//! they stay platform-free.
//!
//! Names are the executable's file name, not its full path — `chrome.exe` on
//! Windows, `Google Chrome` on macOS, `chrome` on Linux. That matches the
//! per-OS tables in [`crate::browser`] and what users type into an app rule.

/// A running process, reduced to what blocking decisions need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Process {
    pub pid: u32,
    pub name: String,
}

impl Process {
    /// Whether terminating this process is something we should ever do.
    ///
    /// Rules can match by name, and a name can collide with the kernel, init,
    /// or Focuser itself. Killing any of those is never what the user meant.
    pub fn is_killable(&self) -> bool {
        self.pid > LOWEST_USER_PID && self.pid != std::process::id()
    }
}

/// PIDs at or below this belong to the kernel or init on every platform we
/// target (0 and 4 on Windows, 0 and 1 on Unix).
const LOWEST_USER_PID: u32 = 4;

/// Every process currently running.
///
/// Returns an empty list rather than an error when the OS refuses to answer —
/// the callers poll on a timer, so a failed sweep should skip a beat, not
/// bring down the loop.
pub fn list() -> Vec<Process> {
    imp::list()
}

/// Ask a process to exit. Returns whether the request was delivered.
///
/// Unix sends `SIGTERM`, so a process that traps it may survive; Windows
/// terminates outright. Neither waits for the process to actually go away.
pub fn terminate(pid: u32) -> bool {
    imp::terminate(pid)
}

/// The full command line a process was started with, if it can be read.
///
/// Call it for specific suspects, never across a whole process list: macOS
/// starts `ps` for every read. Windows and Linux ask the kernel.
pub fn cmdline(pid: u32) -> Option<String> {
    imp::cmdline(pid)
}

/// The name [`list`] would report for the program at `path`.
///
/// A rule stores a name and blocking compares it against a running process, so
/// picking a file from disk has to land on the same string the OS will hand
/// back later. Each platform mangles it differently and none of them is the
/// plain file name.
pub fn name_for_path(path: &str) -> String {
    imp::name_for_path(path)
}

/// The name [`list`] would report for one running process, without listing
/// them all.
///
/// The foreground watcher knows a pid and needs the name a rule is written
/// against. Going through the same per-OS source as [`list`] is what keeps an
/// allowance and a block agreeing about what the app is called.
#[cfg(not(windows))]
pub fn name_for_pid(pid: u32) -> Option<String> {
    imp::name_for_pid(pid)
}

/// The trailing path component, for the platforms where that is the whole job.
fn file_name(path: &str) -> String {
    let trimmed = path.trim_end_matches(['/', '\\']);
    trimmed
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(trimmed)
        .to_string()
}

#[cfg(windows)]
mod imp {
    use super::Process;
    use windows::Wdk::System::Threading::{
        NtQueryInformationProcess, ProcessCommandLineInformation,
    };
    use windows::Win32::Foundation::{CloseHandle, HANDLE, UNICODE_STRING};
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW,
        TH32CS_SNAPPROCESS,
    };
    use windows::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE, TerminateProcess,
    };

    pub fn list() -> Vec<Process> {
        let mut found = Vec::new();

        // SAFETY: the snapshot handle is checked before use and closed on every
        // path out; `entry` is sized per the Win32 contract before the walk.
        unsafe {
            let Ok(snapshot) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else {
                return found;
            };

            let mut entry = PROCESSENTRY32W {
                dwSize: size_of::<PROCESSENTRY32W>() as u32,
                ..Default::default()
            };

            if Process32FirstW(snapshot, &mut entry).is_ok() {
                loop {
                    let len = entry
                        .szExeFile
                        .iter()
                        .position(|&c| c == 0)
                        .unwrap_or(entry.szExeFile.len());

                    found.push(Process {
                        pid: entry.th32ProcessID,
                        name: String::from_utf16_lossy(&entry.szExeFile[..len]),
                    });

                    if Process32NextW(snapshot, &mut entry).is_err() {
                        break;
                    }
                }
            }

            let _ = CloseHandle(snapshot);
        }

        found
    }

    /// Asks the kernel directly. This used to start PowerShell and query WMI,
    /// about a second per process. The uninstall watch reads every shell on
    /// the machine, so on a desktop with thirty of them open the blocking loop
    /// stood still for half a minute, and the app with it.
    pub fn cmdline(pid: u32) -> Option<String> {
        // SAFETY: the handle is only used when OpenProcess succeeded, and is
        // closed before returning.
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
            let line = command_line_of(handle);
            let _ = CloseHandle(handle);
            line
        }
    }

    /// `ProcessCommandLineInformation` (Windows 8.1 and later) fills the buffer
    /// with a `UNICODE_STRING` whose text follows it in the same buffer.
    ///
    /// SAFETY: `handle` must be an open process handle with query access.
    unsafe fn command_line_of(handle: HANDLE) -> Option<String> {
        // Asked with no buffer, it fails and reports the size it needs.
        let mut needed = 0u32;
        let _ = unsafe {
            NtQueryInformationProcess(
                handle,
                ProcessCommandLineInformation,
                std::ptr::null_mut(),
                0,
                &mut needed,
            )
        };
        if (needed as usize) < size_of::<UNICODE_STRING>() {
            return None;
        }

        // u64s, so the UNICODE_STRING at the front is aligned for its pointer.
        let mut buffer = vec![0u64; (needed as usize).div_ceil(size_of::<u64>())];
        let bytes = buffer.len() * size_of::<u64>();
        let status = unsafe {
            NtQueryInformationProcess(
                handle,
                ProcessCommandLineInformation,
                buffer.as_mut_ptr().cast(),
                bytes as u32,
                &mut needed,
            )
        };
        if status.is_err() {
            return None;
        }

        // SAFETY: the call succeeded, so the buffer starts with the header it
        // wrote, and the buffer is aligned and large enough for one.
        let header = unsafe { &*buffer.as_ptr().cast::<UNICODE_STRING>() };
        let text = header.Buffer.0 as usize;
        let length = header.Length as usize;
        let start = buffer.as_ptr() as usize;
        // The text must lie inside the buffer we own. Anything else is not
        // something to read from.
        if text < start || text + length > start + bytes || !text.is_multiple_of(align_of::<u16>())
        {
            return None;
        }
        // SAFETY: checked just above to be inside `buffer`, which is alive and
        // not written to while this slice exists. `Length` is in bytes.
        let units = unsafe { std::slice::from_raw_parts(header.Buffer.0, length / 2) };

        let line = String::from_utf16_lossy(units).trim().to_string();
        (!line.is_empty()).then_some(line)
    }

    pub fn terminate(pid: u32) -> bool {
        // SAFETY: the handle is only used when OpenProcess succeeded, and is
        // closed before returning.
        unsafe {
            let Ok(handle) = OpenProcess(PROCESS_TERMINATE, false, pid) else {
                return false;
            };
            let killed = TerminateProcess(handle, 1).is_ok();
            let _ = CloseHandle(handle);
            killed
        }
    }

    /// `szExeFile` is the file name, extension and all, so there is nothing to
    /// resolve here.
    pub fn name_for_path(path: &str) -> String {
        super::file_name(path)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::os::windows::process::CommandExt;
        use std::time::{Duration, Instant};
        use windows::Win32::System::Threading::CREATE_NO_WINDOW;

        #[test]
        fn a_command_line_is_read_from_another_process() {
            // What the uninstall watch does: a shell it did not start, by pid.
            let mut shell = std::process::Command::new("cmd")
                .args([
                    "/c",
                    "ping -n 6 127.0.0.1 >nul & rem focuser-cmdline-marker",
                ])
                .creation_flags(CREATE_NO_WINDOW.0)
                .spawn()
                .unwrap();
            let line = cmdline(shell.id());
            let _ = shell.kill();
            let _ = shell.wait();

            let line = line.expect("a running shell has a command line");
            assert!(line.contains("focuser-cmdline-marker"), "{line}");
        }

        #[test]
        fn reading_a_command_line_does_not_start_a_program() {
            // At a second per read this would take over three minutes.
            let started = Instant::now();
            for _ in 0..200 {
                assert!(cmdline(std::process::id()).is_some());
            }
            assert!(started.elapsed() < Duration::from_secs(2));
        }

        #[test]
        fn a_process_that_is_not_there_has_no_command_line() {
            assert_eq!(cmdline(u32::MAX - 3), None);
        }
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use super::Process;

    pub fn list() -> Vec<Process> {
        // `comm` is the full executable path on macOS, so take the last
        // component: `.../Google Chrome.app/Contents/MacOS/Google Chrome`
        // has to match the `Google Chrome` in the browser table.
        let Ok(output) = std::process::Command::new("ps")
            .args(["-eo", "pid=,comm="])
            .output()
        else {
            return Vec::new();
        };

        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(parse_ps_line)
            .collect()
    }

    /// One `pid=,comm=` row: leading-padded pid, a space, then the path.
    fn parse_ps_line(line: &str) -> Option<Process> {
        let trimmed = line.trim_start();
        let split = trimmed.find(' ')?;
        let pid = trimmed[..split].parse().ok()?;
        let path = trimmed[split..].trim();
        if path.is_empty() {
            return None;
        }
        Some(Process {
            pid,
            name: path.rsplit('/').next().unwrap_or(path).to_string(),
        })
    }

    pub fn cmdline(pid: u32) -> Option<String> {
        let output = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "args="])
            .output()
            .ok()?;
        let line = String::from_utf8_lossy(&output.stdout).trim().to_string();
        (!line.is_empty()).then_some(line)
    }

    pub fn terminate(pid: u32) -> bool {
        super::unix_terminate(pid)
    }

    /// `ps comm=` reports the binary *inside* the bundle, so picking
    /// `Google Chrome.app` has to resolve to `Google Chrome`. The name is
    /// usually the bundle's, but `CFBundleExecutable` is the only place that
    /// actually says so.
    pub fn name_for_path(path: &str) -> String {
        let trimmed = path.trim_end_matches('/');
        if !trimmed.ends_with(".app") {
            return super::file_name(trimmed);
        }

        let bundle = std::path::Path::new(trimmed);
        declared_executable(&bundle.join("Contents/Info.plist"))
            .unwrap_or_else(|| super::file_name(trimmed.trim_end_matches(".app")))
    }

    /// `ps -p` rather than a full sweep: the caller already knows the pid and
    /// runs on a timer.
    pub fn name_for_pid(pid: u32) -> Option<String> {
        let output = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "comm="])
            .output()
            .ok()?;
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        (!path.is_empty()).then(|| super::file_name(&path))
    }

    fn declared_executable(info_plist: &std::path::Path) -> Option<String> {
        let value = plist::Value::from_file(info_plist).ok()?;
        let name = value
            .as_dictionary()?
            .get("CFBundleExecutable")?
            .as_string()?
            .trim()
            .to_string();
        (!name.is_empty()).then_some(name)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_bundle_resolves_to_its_declared_executable() {
            let dir = tempfile::tempdir().unwrap();
            let bundle = dir.path().join("Google Chrome.app");
            let contents = bundle.join("Contents");
            std::fs::create_dir_all(&contents).unwrap();
            std::fs::write(
                contents.join("Info.plist"),
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>Google Chrome</string>
</dict></plist>"#,
            )
            .unwrap();

            let name = name_for_path(bundle.to_str().unwrap());
            assert_eq!(name, "Google Chrome");
        }

        #[test]
        fn a_bundle_without_a_plist_falls_back_to_its_own_name() {
            let dir = tempfile::tempdir().unwrap();
            let bundle = dir.path().join("Slack.app");
            std::fs::create_dir_all(&bundle).unwrap();
            assert_eq!(name_for_path(bundle.to_str().unwrap()), "Slack");
        }

        #[test]
        fn a_bare_binary_keeps_its_file_name() {
            assert_eq!(name_for_path("/usr/local/bin/node"), "node");
        }

        #[test]
        fn a_ps_row_yields_the_executable_basename() {
            let row = "  842 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
            let proc = parse_ps_line(row).expect("row should parse");
            assert_eq!(proc.pid, 842);
            assert_eq!(proc.name, "Google Chrome");
        }

        #[test]
        fn a_bare_command_name_is_left_alone() {
            assert_eq!(parse_ps_line("1 launchd").unwrap().name, "launchd");
        }

        #[test]
        fn rows_without_a_pid_or_a_command_are_skipped() {
            assert!(parse_ps_line("  PID COMM").is_none());
            assert!(parse_ps_line("842 ").is_none());
            assert!(parse_ps_line("").is_none());
        }
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::Process;

    pub fn list() -> Vec<Process> {
        let Ok(entries) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };

        entries
            .flatten()
            .filter_map(|entry| {
                let pid = entry.file_name().to_string_lossy().parse().ok()?;
                // `comm` is the 15-char command name the kernel tracks, which
                // is what the Linux browser table is written against.
                let comm = std::fs::read_to_string(entry.path().join("comm")).ok()?;
                let name = comm.trim();
                if name.is_empty() {
                    return None;
                }
                Some(Process {
                    pid,
                    name: name.to_string(),
                })
            })
            .collect()
    }

    pub fn cmdline(pid: u32) -> Option<String> {
        // /proc separates argv entries with NULs and usually trails one.
        let raw = std::fs::read_to_string(format!("/proc/{pid}/cmdline")).ok()?;
        let line = raw.replace(' ', " ").trim().to_string();
        (!line.is_empty()).then_some(line)
    }

    pub fn terminate(pid: u32) -> bool {
        super::unix_terminate(pid)
    }

    /// The kernel stores `comm` in 16 bytes including the terminator, so
    /// `list` reports at most 15 characters. A rule holding the full file name
    /// of a longer binary would never match a running one.
    pub const COMM_LEN: usize = 15;

    pub fn name_for_path(path: &str) -> String {
        truncate_to_comm(&super::file_name(path))
    }

    /// Straight from `comm`, the same file `list` reads, so no truncation is
    /// needed here — the kernel has already done it.
    pub fn name_for_pid(pid: u32) -> Option<String> {
        let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).ok()?;
        let name = comm.trim();
        (!name.is_empty()).then(|| name.to_string())
    }

    fn truncate_to_comm(name: &str) -> String {
        match name.char_indices().nth(COMM_LEN) {
            Some((cut, _)) => name[..cut].to_string(),
            None => name.to_string(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_short_name_is_left_alone() {
            assert_eq!(name_for_path("/usr/bin/firefox"), "firefox");
        }

        #[test]
        fn a_long_name_is_cut_where_the_kernel_cuts_it() {
            let name = name_for_path("/opt/bin/some-very-long-binary");
            assert_eq!(name, "some-very-long-");
            assert_eq!(name.chars().count(), COMM_LEN);
        }

        #[test]
        fn truncation_never_splits_a_character() {
            let cut = truncate_to_comm("ααααααααααααααααα");
            assert_eq!(cut.chars().count(), COMM_LEN);
        }
    }
}

#[cfg(unix)]
fn unix_terminate(pid: u32) -> bool {
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    kill(Pid::from_raw(pid as i32), Signal::SIGTERM).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn our_own_process_is_never_killable() {
        let me = Process {
            pid: std::process::id(),
            name: "focuser".into(),
        };
        assert!(!me.is_killable());
    }

    #[test]
    fn kernel_and_init_pids_are_never_killable() {
        for pid in 0..=LOWEST_USER_PID {
            let proc = Process {
                pid,
                name: "system".into(),
            };
            assert!(!proc.is_killable(), "pid {pid} should be protected");
        }
    }

    #[test]
    fn an_ordinary_pid_is_killable() {
        let other = std::process::id() + 1000;
        assert!(
            Process {
                pid: other,
                name: "chrome.exe".into()
            }
            .is_killable()
        );
    }

    #[test]
    fn listing_finds_the_test_binary_itself() {
        let running = list();
        assert!(!running.is_empty(), "the OS should report some processes");
        assert!(
            running.iter().any(|p| p.pid == std::process::id()),
            "our own pid should appear in the list"
        );
    }
}
