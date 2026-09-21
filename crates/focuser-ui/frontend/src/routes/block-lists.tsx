import { Keyboard, ListChecks, Lock, Plus, Trash2 } from "lucide-react";
import { type FormEvent, useId, useState } from "react";
import type { BlockList, ProtectionInfo, TypingLockInfo } from "@/bindings";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, EmptyState, PageHeader } from "@/components/ui/card";
import { InlineError, QueryState } from "@/components/ui/feedback";
import { Input } from "@/components/ui/input";
import { NumberField } from "@/components/ui/number-field";
import { Page } from "@/components/ui/page";
import { ListSkeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Tabs } from "@/components/ui/tabs";
import { Tooltip } from "@/components/ui/tooltip";
import {
  useAttemptUnlock,
  useBlockLists,
  useCreateBlockList,
  useDeleteBlockList,
  useEnableProtection,
  useEnableTypingLock,
  useProtectionStatus,
  useRequestUnlockPhrase,
  useToggleBlockList,
  useTypingLockStatus,
} from "@/lib/commands";
import { formatDuration } from "@/lib/duration";
import { m } from "@/paraglide/messages.js";

export function BlockLists() {
  const [name, setName] = useState("");
  const lists = useBlockLists();
  const protection = useProtectionStatus();
  const typingLockStatus = useTypingLockStatus();
  const create = useCreateBlockList();

  const locks = protection.data ?? [];
  const typingLocks = typingLockStatus.data ?? [];

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    create.mutate(name, { onSuccess: () => setName("") });
  }

  return (
    <Page>
      <PageHeader title={m.lists_title()} description={m.lists_description()} />

      <Card className="mb-6" padding="md" elevation="raised">
        <form onSubmit={onSubmit} className="flex gap-2">
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={m.lists_new_placeholder()}
            aria-label={m.lists_new_label()}
          />
          <Button type="submit" icon={<Plus />} disabled={!name.trim() || create.isPending}>
            {create.isPending ? m.lists_creating() : m.lists_create()}
          </Button>
        </form>
        <InlineError error={create.error} />
      </Card>

      {lists.isPending ? (
        <ListSkeleton rows={3} />
      ) : (
        <QueryState
          isPending={false}
          error={lists.error}
          onRetry={() => lists.refetch()}
          isRetrying={lists.isFetching}
        >
          {lists.data?.length === 0 ? (
            <EmptyState
              icon={<ListChecks />}
              title={m.lists_empty_title()}
              description={m.lists_empty_description()}
            />
          ) : (
            <ul className="flex flex-col gap-2">
              {lists.data?.map((list) => (
                <ListRow
                  key={list.id}
                  list={list}
                  lock={locks.find((p) => p.block_list_id === list.id) ?? null}
                  typingLock={typingLocks.find((t) => t.block_list_id === list.id) ?? null}
                />
              ))}
            </ul>
          )}
        </QueryState>
      )}
    </Page>
  );
}

function ListRow({
  list,
  lock,
  typingLock,
}: {
  list: BlockList;
  lock: ProtectionInfo | null;
  typingLock: TypingLockInfo | null;
}) {
  const toggle = useToggleBlockList();
  const remove = useDeleteBlockList();
  const [showLockForm, setShowLockForm] = useState(false);
  const [showUnlock, setShowUnlock] = useState(false);

  const held = lock !== null || typingLock !== null;

  return (
    <li>
      <Card data-testid="block-list-row" elevation="raised" padding="none">
        <div className="flex items-center justify-between gap-4 px-4 py-3.5">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="truncate font-medium text-foreground text-sm">{list.name}</p>
              {lock ? (
                <Badge tone="warning" icon={<Lock aria-hidden />} outlined>
                  {m.lists_badge_locked({ duration: formatDuration(lock.remaining_seconds) })}
                </Badge>
              ) : typingLock ? (
                <Badge tone="warning" icon={<Keyboard aria-hidden />} outlined>
                  {m.lists_badge_typing_locked({ length: typingLock.phrase_length })}
                </Badge>
              ) : (
                <Badge tone={list.enabled ? "success" : "neutral"}>
                  {list.enabled ? m.lists_badge_enabled() : m.lists_badge_off()}
                </Badge>
              )}
              {list.schedule && <Badge tone="info">{m.lists_badge_scheduled()}</Badge>}
            </div>
            <p className="mt-1 text-faint-foreground text-xs">
              {m.count_sites({ count: list.websites.length })} ·{" "}
              {m.count_apps({ count: list.applications.length })} ·{" "}
              {m.count_exceptions({ count: list.exceptions.length })}
            </p>
          </div>

          <div className="flex shrink-0 items-center gap-1">
            <Tooltip content={held ? m.lists_toggle_locked() : m.lists_toggle_hint()}>
              <span>
                <Switch
                  checked={list.enabled}
                  onCheckedChange={(enabled) => toggle.mutate({ id: list.id, enabled })}
                  disabled={held && list.enabled}
                  aria-label={m.lists_enable({ name: list.name })}
                />
              </span>
            </Tooltip>

            <Tooltip content={held ? m.lists_protect_already() : m.lists_protect_hint()}>
              <span>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={m.lists_protect({ name: list.name })}
                  aria-expanded={showLockForm}
                  disabled={held}
                  onClick={() => setShowLockForm(!showLockForm)}
                >
                  <Lock />
                </Button>
              </span>
            </Tooltip>

            {typingLock && (
              <Tooltip content={m.lists_unlock_hint()}>
                <span>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={m.lists_unlock({ name: list.name })}
                    aria-expanded={showUnlock}
                    onClick={() => setShowUnlock(!showUnlock)}
                  >
                    <Keyboard />
                  </Button>
                </span>
              </Tooltip>
            )}

            <Tooltip content={held ? m.lists_delete_locked() : m.lists_delete_hint()}>
              <span>
                <Button
                  variant="ghost"
                  tone="destructive"
                  size="icon"
                  aria-label={m.lists_delete({ name: list.name })}
                  disabled={held}
                  onClick={() => remove.mutate(list.id)}
                >
                  <Trash2 />
                </Button>
              </span>
            </Tooltip>
          </div>
        </div>

        {showLockForm && !held && <LockForm list={list} onDone={() => setShowLockForm(false)} />}
        {showUnlock && typingLock && <UnlockForm list={list} onDone={() => setShowUnlock(false)} />}

        <InlineError error={toggle.error ?? remove.error} />
      </Card>
    </li>
  );
}

/** Choose how to lock the list: a countdown timer, or a typed phrase. */
function LockForm({ list, onDone }: { list: BlockList; onDone: () => void }) {
  const [mode, setMode] = useState<"timer" | "typing">("timer");

  return (
    <div className="animate-in border-border border-t bg-elevated/40 px-4 py-4 fade-in slide-in-from-top-1">
      <p className="flex items-center gap-2 font-medium text-foreground text-sm">
        <Lock aria-hidden className="size-4 text-warning" />
        {m.lists_lock_heading()}
      </p>

      <Tabs
        className="mt-3"
        value={mode}
        onChange={setMode}
        items={[
          { id: "timer", label: m.lists_lock_mode_timer() },
          { id: "typing", label: m.lists_lock_mode_typing() },
        ]}
      />

      {mode === "timer" ? (
        <TimerLockFields list={list} onDone={onDone} />
      ) : (
        <TypingLockFields list={list} onDone={onDone} />
      )}
    </div>
  );
}

function TimerLockFields({ list, onDone }: { list: BlockList; onDone: () => void }) {
  const protect = useEnableProtection();
  const id = useId();

  const [minutes, setMinutes] = useState(60);
  const [uninstall, setUninstall] = useState(true);
  const [serviceStop, setServiceStop] = useState(true);
  const [modification, setModification] = useState(true);

  return (
    <div className="mt-4">
      <p className="text-muted-foreground text-sm">{m.lists_lock_warning()}</p>

      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex items-center gap-2">
          <label htmlFor={id} className="text-muted-foreground text-sm">
            {m.lists_lock_for()}
          </label>
          <NumberField
            id={id}
            value={minutes}
            onCommit={setMinutes}
            min={1}
            max={10080}
            suffix="min"
          />
        </div>

        <Guard label={m.lists_guard_uninstall()} checked={uninstall} onChange={setUninstall} />
        <Guard label={m.lists_guard_service()} checked={serviceStop} onChange={setServiceStop} />
        <Guard label={m.lists_guard_edit()} checked={modification} onChange={setModification} />
      </div>

      <div className="mt-4 flex items-center gap-2">
        <Button
          size="sm"
          tone="destructive"
          disabled={protect.isPending}
          onClick={() =>
            protect.mutate(
              {
                listId: list.id,
                minutes,
                preventUninstall: uninstall,
                preventServiceStop: serviceStop,
                preventModification: modification,
              },
              { onSuccess: onDone },
            )
          }
        >
          {protect.isPending
            ? m.lists_locking()
            : m.lists_lock_action({ duration: formatDuration(minutes * 60) })}
        </Button>
        <Button variant="ghost" size="sm" onClick={onDone}>
          {m.lists_cancel()}
        </Button>
      </div>

      <InlineError error={protect.error} />
    </div>
  );
}

function TypingLockFields({ list, onDone }: { list: BlockList; onDone: () => void }) {
  const lockWithPhrase = useEnableTypingLock();
  const id = useId();
  const [length, setLength] = useState(200);

  return (
    <div className="mt-4">
      <p className="text-muted-foreground text-sm">{m.lists_typing_lock_warning()}</p>

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
          disabled={lockWithPhrase.isPending}
          onClick={() =>
            lockWithPhrase.mutate({ listId: list.id, phraseLength: length }, { onSuccess: onDone })
          }
        >
          {lockWithPhrase.isPending ? m.lists_locking() : m.lists_typing_lock_action({ length })}
        </Button>
        <Button variant="ghost" size="sm" onClick={onDone}>
          {m.lists_cancel()}
        </Button>
      </div>

      <InlineError error={lockWithPhrase.error} />
    </div>
  );
}

/**
 * Type the phrase back to unlock.
 *
 * The phrase is only ever requested on demand (never on mount), so opening
 * and closing this panel casually does not spend it. Pasting is blocked —
 * the point of the lock is the typing itself, not reading the phrase back.
 */
function UnlockForm({ list, onDone }: { list: BlockList; onDone: () => void }) {
  const requestPhrase = useRequestUnlockPhrase();
  const attempt = useAttemptUnlock();
  const [phrase, setPhrase] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const [wrong, setWrong] = useState(false);
  const id = useId();

  function newPhrase() {
    setWrong(false);
    setTyped("");
    requestPhrase.mutate(list.id, { onSuccess: setPhrase });
  }

  function submit() {
    attempt.mutate(
      { listId: list.id, typed },
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
    <div className="animate-in border-border border-t bg-elevated/40 px-4 py-4 fade-in slide-in-from-top-1">
      <p className="flex items-center gap-2 font-medium text-foreground text-sm">
        <Keyboard aria-hidden className="size-4 text-warning" />
        {m.lists_unlock_heading()}
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

function Guard({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <Switch size="sm" checked={checked} onCheckedChange={onChange} aria-label={label} />
      <span className="text-muted-foreground text-sm">{label}</span>
    </div>
  );
}
