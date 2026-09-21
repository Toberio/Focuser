import { Keyboard, Lock } from "lucide-react";
import { useId, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { InlineError } from "@/components/ui/feedback";
import { NumberField } from "@/components/ui/number-field";
import { Tooltip } from "@/components/ui/tooltip";
import {
  useAttemptSettingUnlock,
  useEnableSettingLock,
  useRequestSettingUnlockPhrase,
  useSettingLockStatus,
} from "@/lib/commands";
import { m } from "@/paraglide/messages.js";

/**
 * A padlock for a settings-table key: once locked, the setting cannot be
 * changed until a freshly generated random phrase is typed back exactly —
 * same mechanism as a block list's typing lock (see `routes/block-lists.tsx`),
 * just keyed by a settings key instead of a list id.
 *
 * Returns pieces rather than one component, because the badge and lock/unlock
 * button belong inline in the row's control slot, while the expandable
 * lock/unlock panel needs to render as a sibling below the whole row (a
 * `SettingRow` has no slot for it) — the caller places `panel` there.
 */
export function useSettingLock(settingKey: string, label: string) {
  const status = useSettingLockStatus(settingKey);
  const [showLockForm, setShowLockForm] = useState(false);
  const [showUnlock, setShowUnlock] = useState(false);

  const badge = status.locked ? (
    <Badge tone="warning" icon={<Lock aria-hidden className="size-3" />} outlined size="sm">
      {m.settings_badge_locked({ length: status.phraseLength ?? 0 })}
    </Badge>
  ) : null;

  const button = (
    <Tooltip content={status.locked ? m.settings_lock_already() : m.settings_lock_hint()}>
      <span>
        <Button
          variant="ghost"
          size="icon"
          aria-label={
            status.locked ? m.settings_unlock({ name: label }) : m.settings_lock({ name: label })
          }
          aria-expanded={status.locked ? showUnlock : showLockForm}
          onClick={() => (status.locked ? setShowUnlock((v) => !v) : setShowLockForm((v) => !v))}
        >
          {status.locked ? <Keyboard /> : <Lock />}
        </Button>
      </span>
    </Tooltip>
  );

  const panel =
    showLockForm && !status.locked ? (
      <SettingLockForm settingKey={settingKey} onDone={() => setShowLockForm(false)} />
    ) : showUnlock && status.locked ? (
      <SettingUnlockForm settingKey={settingKey} onDone={() => setShowUnlock(false)} />
    ) : null;

  return { locked: status.locked, badge, button, panel };
}

function SettingLockForm({ settingKey, onDone }: { settingKey: string; onDone: () => void }) {
  const lock = useEnableSettingLock();
  const id = useId();
  const [length, setLength] = useState(200);

  return (
    <div className="animate-in border-border border-t bg-elevated/40 px-5 py-4 fade-in slide-in-from-top-1">
      <p className="flex items-center gap-2 font-medium text-foreground text-sm">
        <Lock aria-hidden className="size-4 text-warning" />
        {m.settings_lock_heading()}
      </p>
      <p className="mt-1 text-muted-foreground text-sm">{m.settings_lock_warning()}</p>

      <div className="mt-4 flex items-center gap-2">
        <label htmlFor={id} className="text-muted-foreground text-sm">
          {m.lists_typing_lock_length()}
        </label>
        <NumberField
          id={id}
          value={length}
          onCommit={setLength}
          min={10}
          max={5000}
          suffix="chars"
        />
      </div>

      <div className="mt-4 flex items-center gap-2">
        <Button
          size="sm"
          tone="destructive"
          disabled={lock.isPending}
          onClick={() =>
            lock.mutate({ key: settingKey, phraseLength: length }, { onSuccess: onDone })
          }
        >
          {lock.isPending ? m.lists_locking() : m.settings_lock_action({ length })}
        </Button>
        <Button variant="ghost" size="sm" onClick={onDone}>
          {m.lists_cancel()}
        </Button>
      </div>

      <InlineError error={lock.error} />
    </div>
  );
}

function SettingUnlockForm({ settingKey, onDone }: { settingKey: string; onDone: () => void }) {
  const requestPhrase = useRequestSettingUnlockPhrase();
  const attempt = useAttemptSettingUnlock();
  const [phrase, setPhrase] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const [wrong, setWrong] = useState(false);
  const id = useId();

  function newPhrase() {
    setWrong(false);
    setTyped("");
    requestPhrase.mutate(settingKey, { onSuccess: setPhrase });
  }

  function submit() {
    attempt.mutate(
      { key: settingKey, typed },
      {
        onSuccess: (matched) => {
          if (matched) {
            onDone();
            return;
          }
          setWrong(true);
          setPhrase(null);
          setTyped("");
        },
      },
    );
  }

  const matchesSoFar = phrase?.startsWith(typed) ?? false;
  const complete = phrase !== null && typed === phrase;

  return (
    <div className="animate-in border-border border-t bg-elevated/40 px-5 py-4 fade-in slide-in-from-top-1">
      <p className="flex items-center gap-2 font-medium text-foreground text-sm">
        <Keyboard aria-hidden className="size-4 text-warning" />
        {m.settings_unlock_heading()}
      </p>
      <p className="mt-1 text-muted-foreground text-sm">{m.lists_unlock_instructions()}</p>

      {phrase === null ? (
        <div className="mt-4">
          <Button size="sm" disabled={requestPhrase.isPending} onClick={newPhrase}>
            {requestPhrase.isPending ? m.lists_unlock_loading() : m.lists_unlock_new_phrase()}
          </Button>
          {wrong && <p className="mt-2 text-destructive text-sm">{m.lists_unlock_wrong()}</p>}
          <div className="mt-2">
            <Button variant="ghost" size="sm" onClick={onDone}>
              {m.lists_cancel()}
            </Button>
          </div>
        </div>
      ) : (
        <div className="mt-4">
          <div
            className="select-none whitespace-pre-wrap break-all rounded-md border border-border bg-surface p-3 font-mono text-sm"
            aria-hidden
          >
            {phrase}
          </div>

          <label htmlFor={id} className="mt-3 block text-muted-foreground text-sm">
            {m.lists_unlock_input_label()}
          </label>
          <textarea
            id={id}
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            onPaste={(e) => e.preventDefault()}
            onDrop={(e) => e.preventDefault()}
            rows={4}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck={false}
            className="mt-1 w-full resize-none rounded-md border border-border bg-surface p-3 font-mono text-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
          />

          <p
            className={`mt-2 text-sm ${
              typed.length === 0
                ? "text-faint-foreground"
                : matchesSoFar
                  ? "text-success"
                  : "text-destructive"
            }`}
          >
            {matchesSoFar || typed.length === 0
              ? m.lists_unlock_progress({ typed: typed.length, total: phrase.length })
              : m.lists_unlock_mismatch()}
          </p>

          <div className="mt-3 flex items-center gap-2">
            <Button
              size="sm"
              tone="destructive"
              disabled={!complete || attempt.isPending}
              onClick={submit}
            >
              {attempt.isPending ? m.lists_unlocking() : m.lists_unlock_submit()}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={newPhrase}
              disabled={requestPhrase.isPending}
            >
              {m.lists_unlock_new_phrase()}
            </Button>
            <Button variant="ghost" size="sm" onClick={onDone}>
              {m.lists_cancel()}
            </Button>
          </div>

          <InlineError error={attempt.error} />
        </div>
      )}
    </div>
  );
}
