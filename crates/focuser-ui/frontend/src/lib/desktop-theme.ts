/**
 * Follow the desktop's theme where it publishes one (Omarchy, for now).
 *
 * The desktop app reads the active palette and this maps it onto the raw
 * tokens in `index.css`, as inline properties on `<html>` so they win over the
 * stylesheet. Re-read on window focus: a theme switch happens from the
 * desktop's own menu, so focus coming back is when it can first be seen.
 */

import { useQuery } from "@tanstack/react-query";
import { invoke } from "@tauri-apps/api/core";
import { useEffect } from "react";
import { isTauri } from "./transport";

export interface DesktopTheme {
  omarchy: boolean;
  mode: "dark" | "light" | null;
  colors: Record<string, string>;
}

const queryKey = ["desktop-theme"] as const;

export function useDesktopTheme(): DesktopTheme | null {
  const theme = useQuery({
    queryKey,
    queryFn: () => invoke<DesktopTheme | null>("desktop_theme"),
    enabled: isTauri(),
    refetchInterval: false,
    refetchOnWindowFocus: true,
  });
  return theme.data ?? null;
}

/** Apply the desktop palette for as long as the caller is mounted. */
export function useApplyDesktopTheme() {
  const theme = useDesktopTheme();

  useEffect(() => {
    const tokens = theme ? tokensFor(theme) : null;
    if (!tokens) return;
    const root = document.documentElement;
    for (const [name, value] of Object.entries(tokens)) root.style.setProperty(name, value);
    root.style.colorScheme = theme?.mode ?? "dark";
    return () => {
      for (const name of Object.keys(tokens)) root.style.removeProperty(name);
      root.style.colorScheme = "";
    };
  }, [theme]);
}

const mix = (a: string, b: string, percentOfB: number) =>
  `color-mix(in srgb, ${a}, ${b} ${percentOfB}%)`;
const alpha = (color: string, percent: number) =>
  `color-mix(in srgb, ${color} ${percent}%, transparent)`;

/** Null when the palette lacks the colours the surfaces are built from. */
function tokensFor({ colors: c, mode }: DesktopTheme): Record<string, string> | null {
  const bg = c.background;
  const fg = c.foreground;
  const accent = c.accent ?? c.blue;
  if (!bg || !fg || !accent) return null;

  const light = mode === "light";
  // Surfaces step towards the foreground, which is lighter on a dark theme
  // and darker on a light one, so the ramp keeps its direction either way.
  const deep = c.darker_background ?? mix(bg, light ? "white" : "black", 30);
  const surface = c.lighter_background ?? mix(bg, fg, 5);

  return {
    "--deep": deep,
    "--base": bg,
    "--surface": surface,
    "--elevated": mix(surface, fg, 5),
    "--hover": mix(surface, fg, 9),
    "--active": mix(surface, fg, 13),

    "--foreground": fg,
    "--muted-foreground": c.light_foreground ?? mix(fg, bg, 30),
    "--faint-foreground": c.muted ?? mix(fg, bg, 55),

    "--primary": accent,
    "--primary-hover": mix(accent, fg, 15),
    "--primary-active": mix(accent, bg, 15),
    "--primary-foreground": light ? bg : deep,
    "--primary-dim": alpha(accent, 14),
    "--primary-glow": alpha(accent, 35),

    "--success": c.green ?? accent,
    "--warning": c.yellow ?? accent,
    "--destructive": c.red ?? accent,
    "--info": c.blue ?? accent,

    "--border": alpha(fg, 8),
    "--border-strong": alpha(fg, 15),
    "--ring": accent,

    // Desktop themes are flat, and the stock violet-cyan-pink wash would tint
    // every one of them purple.
    "--wash-1": "transparent",
    "--wash-2": "transparent",
    "--wash-3": "transparent",

    "--chart-1": accent,
    "--chart-2": c.cyan ?? accent,
    "--chart-3": c.yellow ?? accent,
    "--chart-4": c.magenta ?? accent,
    "--chart-5": c.green ?? accent,
  };
}
