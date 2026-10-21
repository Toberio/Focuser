//! Colours from the desktop's own theme, so the window matches its neighbours.
//!
//! Only Omarchy for now. `omarchy theme set` writes the active palette to
//! `~/.local/state/omarchy/current/theme/colors.toml` as flat `key = "#hex"`
//! lines, and replaces the whole directory on every switch. The UI asks again
//! whenever the window regains focus, which is when a theme switch made from
//! the Omarchy menu becomes visible.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct DesktopTheme {
    /// Omarchy tiles every window and closes them with a keybinding, so the
    /// UI drops its title bar there.
    pub omarchy: bool,
    /// `"dark"` or `"light"`, from the theme's `mode` key.
    pub mode: Option<String>,
    /// Colour name to `#hex`. Anything that is not a hex colour is dropped,
    /// since the values end up in CSS.
    pub colors: BTreeMap<String, String>,
}

#[tauri::command]
pub fn desktop_theme() -> Option<DesktopTheme> {
    let path = omarchy_colors_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    Some(parse_colors(&text))
}

fn omarchy_colors_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home).join(".local/state/omarchy/current/theme/colors.toml");
    path.is_file().then_some(path)
}

fn parse_colors(text: &str) -> DesktopTheme {
    let mut mode = None;
    let mut colors = BTreeMap::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"');
        if key == "mode" {
            if value == "dark" || value == "light" {
                mode = Some(value.to_string());
            }
        } else if is_key(key) && is_hex_color(value) {
            colors.insert(key.to_string(), value.to_string());
        }
    }
    DesktopTheme {
        omarchy: true,
        mode,
        colors,
    }
}

fn is_key(key: &str) -> bool {
    !key.is_empty() && key.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
}

fn is_hex_color(value: &str) -> bool {
    value.strip_prefix('#').is_some_and(|hex| {
        matches!(hex.len(), 3 | 4 | 6 | 8) && hex.bytes().all(|b| b.is_ascii_hexdigit())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_mode_and_hex_colors() {
        let theme =
            parse_colors("mode = \"dark\"\n\naccent = \"#78824b\"\nbackground = \"#222222\"\n");
        assert!(theme.omarchy);
        assert_eq!(theme.mode.as_deref(), Some("dark"));
        assert_eq!(theme.colors["accent"], "#78824b");
        assert_eq!(theme.colors["background"], "#222222");
    }

    #[test]
    fn drops_values_that_are_not_hex_colors() {
        let theme = parse_colors(
            "accent = \"red; background: url(x)\"\nfoo = \"#12345\"\nBad-Key = \"#fff\"\n",
        );
        assert!(theme.colors.is_empty());
    }
}
