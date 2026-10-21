// Prints the GitHub release text for a tag: what the tag message says
// changed, then how to install. Used by release.yml, and by hand to fill in
// the text of an older release.
//
//   node .github/scripts/release-notes.mjs v0.10.0 > notes.md
import { execFileSync } from "node:child_process";

const tag = process.argv[2];
if (!tag) {
  console.error("usage: node .github/scripts/release-notes.mjs <tag>");
  process.exit(1);
}

const message = execFileSync("git", ["tag", "-l", "--format=%(contents:body)", tag], {
  encoding: "utf8",
}).trimEnd();

// Git messages are wrapped at 72 columns, and GitHub shows every line end as
// a break. Join the lines of each paragraph and each list item back up.
const lines = [];
for (const line of message.split("\n")) {
  const last = lines.length - 1;
  const continues =
    line.trim() !== "" && !line.startsWith("- ") && last >= 0 && lines[last].trim() !== "";
  if (continues) lines[last] += ` ${line.trim()}`;
  else lines.push(line);
}
const changes = lines.join("\n").trim();

const install = `## Downloads

| Platform | Installer | Portable |
|----------|-----------|----------|
| Windows | \`.exe\` (NSIS installer) | \`Focuser-Portable.exe\` |
| macOS (Intel) | \`.dmg\` | none |
| macOS (Apple Silicon) | \`.dmg\` | none |
| Linux | \`.deb\` / \`.rpm\` / \`.AppImage\` | \`Focuser-Linux-x86_64\` |

## Installation

- **Windows**: Download the \`.exe\` installer, run it, and follow the setup wizard. Focuser will be added to Start Menu and auto-start on boot.
- **macOS**: Download the \`.dmg\`, open it, drag Focuser to Applications.
- **Linux**: Use the \`.deb\` for Debian/Ubuntu (\`sudo apt install ./Focuser_*.deb\`), the \`.rpm\` for Fedora (\`sudo dnf install ./Focuser-*.rpm\`), or the \`.AppImage\` for any distro. If the AppImage shows a white screen, \`Focuser-Linux-x86_64\` runs on your system's own \`webkit2gtk-4.1\` instead.

## Browser extension

- **Chrome / Edge / Brave / Opera**: [Chrome Web Store](https://chromewebstore.google.com/detail/jpnhbpbcmagoonmaleppldmcnaibkbmj)
- **Firefox**: [Firefox Add-ons](https://addons.mozilla.org/en-US/firefox/addon/focuser-website-blocker/)

See the [README](https://github.com/aadeshrao123/Focuser#readme) for full details.`;

console.log(changes ? `## What's new\n\n${changes}\n\n${install}` : install);
