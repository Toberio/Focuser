import { useMutation } from "@tanstack/react-query";
import { FolderOpen, Play, RotateCcw } from "lucide-react";
import { useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { BrowserStatusList } from "@/components/browser-status";
import { ConfigTransfer } from "@/components/config-transfer";
import { SettingRow, SettingsSection } from "@/components/setting-row";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/ui/card";
import { InlineError } from "@/components/ui/feedback";
import { NumberField } from "@/components/ui/number-field";
import { Page } from "@/components/ui/page";
import { Select } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { UpdateCheck } from "@/components/update-check";
import { useAutostart } from "@/lib/autostart";
import {
  useAppVersion,
  useProtectionStatus,
  useSetSetting,
  useSetStatsRetention,
  useSetting,
  useStatsRetention,
} from "@/lib/commands";
import { useLanguage } from "@/lib/language";
import { isTauri, pickSound, previewSound } from "@/lib/native";
import {
  MAX_RETENTION_DAYS,
  SETTING_KEYS,
  useBooleanSetting,
  useNumberSetting,
} from "@/lib/settings";
import { m } from "@/paraglide/messages.js";

export function Settings() {
  const autostart = useAutostart();
  const enforceBrowsers = useBooleanSetting(SETTING_KEYS.blockUnsupportedBrowsers, true);
  const gracePeriod = useNumberSetting(SETTING_KEYS.extensionGracePeriod, 60);
  const phaseSound = useBooleanSetting(SETTING_KEYS.phaseSound, false);
  const soundVolume = useNumberSetting(SETTING_KEYS.phaseSoundVolume, 70);
  const language = useLanguage();

  // A lock can tighten these but never loosen them (#18).
  const locks = useProtectionStatus().data ?? [];
  const editLocked = locks.some((l) => l.prevent_modification);
  const autostartLocked = autostart.value && locks.some((l) => l.prevent_service_stop);
  const browsersLocked = editLocked && enforceBrowsers.value;

  const retention = useStatsRetention();
  const setRetention = useSetStatsRetention();
  const version = useAppVersion();

  // The sidebar badge links here promising the update button, so find it.
  const [params] = useSearchParams();
  const highlightUpdates = params.get("highlight") === "updates";
  const updatesRow = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (highlightUpdates) updatesRow.current?.scrollIntoView({ block: "center" });
  }, [highlightUpdates]);

  return (
    // One column, not two. Splitting settings left/right meant a setting's
    // position on the page carried no meaning — you had to scan both sides.
    <Page>
      <PageHeader title={m.settings_title()} description={m.settings_description()} />

      <SettingsSection title={m.settings_section_startup()}>
        <SettingRow
          label={m.settings_autostart()}
          description={
            autostartLocked
              ? m.settings_locked()
              : autostart.needsAdmin
                ? m.settings_autostart_pending()
                : autostart.supported
                  ? m.settings_autostart_description()
                  : m.settings_autostart_unsupported()
          }
          control={
            <Switch
              checked={autostart.value}
              onCheckedChange={autostart.set}
              disabled={
                !autostart.supported || autostart.isPending || autostart.isSaving || autostartLocked
              }
              aria-label={m.settings_autostart()}
            />
          }
        />
      </SettingsSection>

      <SettingsSection
        title={m.settings_section_browsers()}
        description={m.settings_browsers_description()}
      >
        <SettingRow
          label={m.settings_close_browsers()}
          description={
            browsersLocked ? m.settings_locked() : m.settings_close_browsers_description()
          }
          control={
            <Switch
              checked={enforceBrowsers.value}
              onCheckedChange={enforceBrowsers.set}
              disabled={enforceBrowsers.isPending || enforceBrowsers.isSaving || browsersLocked}
              aria-label={m.settings_close_browsers()}
            />
          }
        />
        <SettingRow
          label={m.settings_grace_period()}
          htmlFor="grace-period"
          description={m.settings_grace_period_description()}
          control={
            <NumberField
              id="grace-period"
              value={gracePeriod.value}
              onCommit={gracePeriod.set}
              min={5}
              max={editLocked ? gracePeriod.value : 3600}
              step={5}
              suffix={m.settings_seconds_suffix()}
              disabled={!enforceBrowsers.value || gracePeriod.isPending}
            />
          }
        />
      </SettingsSection>

      <SettingsSection
        title={m.settings_section_extension()}
        description={`${m.settings_extension_description()} ${m.settings_extension_chromium()}`}
        flush
      >
        <BrowserStatusList />
      </SettingsSection>

      <SettingsSection title={m.settings_section_focus()}>
        <SettingRow
          label={m.settings_sound()}
          description={m.settings_sound_description()}
          control={
            <Switch
              checked={phaseSound.value}
              onCheckedChange={phaseSound.set}
              disabled={phaseSound.isPending || phaseSound.isSaving}
              aria-label={m.settings_sound()}
            />
          }
        />
        <SettingRow
          label={m.settings_sound_volume()}
          htmlFor="sound-volume"
          control={
            <NumberField
              id="sound-volume"
              value={soundVolume.value}
              onCommit={soundVolume.set}
              min={0}
              max={100}
              step={10}
              suffix="%"
              disabled={soundVolume.isPending}
            />
          }
        />
        <SoundFile />
      </SettingsSection>

      <SettingsSection title={m.settings_section_data()}>
        <SettingRow
          label={m.settings_retention()}
          htmlFor="retention"
          description={m.settings_retention_description()}
          control={
            <NumberField
              id="retention"
              value={retention.data ?? 30}
              onCommit={(days) => setRetention.mutate(days)}
              min={1}
              max={MAX_RETENTION_DAYS}
              suffix={m.settings_days_suffix()}
              disabled={retention.isPending}
            />
          }
        />
        <SettingRow
          label={m.settings_config_file()}
          description={m.settings_config_file_description()}
          control={<ConfigTransfer />}
        />
      </SettingsSection>

      <SettingsSection title={m.settings_section_language()}>
        <SettingRow
          label={m.settings_language()}
          htmlFor="language"
          description={m.settings_language_description()}
          control={
            <Select
              id="language"
              value={language.value}
              onValueChange={language.set}
              options={language.options}
              size="sm"
              disabled={language.isPending}
              aria-label={m.settings_language()}
            />
          }
        />
      </SettingsSection>

      <SettingsSection title={m.settings_section_about()}>
        <SettingRow label={m.settings_version()} control={<Version value={version.data} />} />
        <div ref={updatesRow}>
          <SettingRow
            label={m.settings_updates()}
            control={<UpdateCheck />}
            highlight={highlightUpdates}
          />
        </div>
      </SettingsSection>

      <InlineError
        error={
          autostart.error ??
          enforceBrowsers.error ??
          gracePeriod.error ??
          phaseSound.error ??
          soundVolume.error ??
          setRetention.error
        }
      />
    </Page>
  );
}

/** The chime file: built in unless the user picked one. Desktop only. */
function SoundFile() {
  const saved = useSetting(SETTING_KEYS.phaseSoundFile, "");
  const save = useSetSetting();
  const preview = useMutation({ mutationFn: previewSound });
  const path = saved.data ?? "";
  const native = isTauri();

  const choose = async () => {
    const picked = await pickSound();
    if (picked) save.mutate({ key: SETTING_KEYS.phaseSoundFile, value: picked });
  };

  return (
    <div>
      <SettingRow
        label={m.settings_sound_file()}
        description={path ? path.split(/[\\/]/).pop() : m.settings_sound_builtin()}
        control={
          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              icon={<FolderOpen />}
              onClick={choose}
              disabled={!native || save.isPending}
            >
              {m.settings_sound_choose()}
            </Button>
            {path && (
              <Button
                variant="ghost"
                size="sm"
                icon={<RotateCcw />}
                onClick={() => save.mutate({ key: SETTING_KEYS.phaseSoundFile, value: "" })}
                disabled={save.isPending}
              >
                {m.settings_sound_builtin_action()}
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              icon={<Play />}
              onClick={() => preview.mutate()}
              disabled={!native || preview.isPending}
            >
              {m.settings_sound_preview()}
            </Button>
          </div>
        }
      />
      <div className="px-5">
        <InlineError error={save.error ?? preview.error} />
      </div>
    </div>
  );
}

function Version({ value }: { value?: string }) {
  return (
    <span className="text-muted-foreground text-sm tabular-nums">
      {value ? `Focuser ${value}` : "—"}
    </span>
  );
}
