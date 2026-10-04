import {
  Cpu,
  EyeOff,
  HardDriveDownload,
  Lock,
  Plus,
  ShieldCheck,
  TriangleAlert,
  X,
} from "lucide-react";
import { type FormEvent, useState } from "react";
import type { BlockList, ImageFilter, ImageFilterStatus } from "@/bindings";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, EmptyState, PageHeader } from "@/components/ui/card";
import { InlineError, QueryState } from "@/components/ui/feedback";
import { Input } from "@/components/ui/input";
import { Page } from "@/components/ui/page";
import { Progress } from "@/components/ui/progress";
import { Select } from "@/components/ui/select";
import {
  useBlockLists,
  useImageFilterStatus,
  useProtectionStatus,
  useSetImageFilter,
  useSetImageFilterExceptions,
} from "@/lib/commands";
import { m } from "@/paraglide/messages.js";

const LEVELS = ["off", "explicit", "balanced", "strict"] as const satisfies readonly ImageFilter[];

/** Called during render, so the labels follow the current language. */
function levelLabel(level: ImageFilter): { label: string; hint: string } {
  switch (level) {
    case "off":
      return { label: m.images_level_off(), hint: m.images_level_off_hint() };
    case "explicit":
      return { label: m.images_level_explicit(), hint: m.images_level_explicit_hint() };
    case "balanced":
      return { label: m.images_level_balanced(), hint: m.images_level_balanced_hint() };
    case "strict":
      return { label: m.images_level_strict(), hint: m.images_level_strict_hint() };
  }
}

/**
 * The explicit-image filter: what it is doing, and how strict each list is.
 *
 * Its own page rather than a corner of Websites, because it is not a list of
 * sites: it judges what is on any page, and has models to download and a
 * state worth seeing.
 */
export function Images() {
  const lists = useBlockLists();
  const protection = useProtectionStatus();
  const all = lists.data ?? [];
  const locked = new Set((protection.data ?? []).map((p) => p.block_list_id));

  return (
    <Page>
      <PageHeader title={m.images_title()} description={m.images_description()} />

      <FilterStatus />

      <section className="mb-7">
        <h2 className="font-semibold text-foreground text-sm">{m.images_lists_title()}</h2>
        <p className="mt-1 text-muted-foreground text-sm">{m.images_lists_description()}</p>
        <QueryState
          isPending={lists.isPending}
          error={lists.error}
          onRetry={() => lists.refetch()}
          isRetrying={lists.isFetching}
        >
          {all.length === 0 ? (
            <EmptyState
              icon={<EyeOff />}
              title={m.images_no_lists_title()}
              description={m.images_no_lists_description()}
            />
          ) : (
            <Card className="mt-3 divide-y divide-border" padding="none" elevation="raised">
              {all.map((list) => (
                <ListLevel key={list.id} list={list} locked={locked.has(list.id)} />
              ))}
            </Card>
          )}
        </QueryState>
      </section>

      <HowItWorks />
    </Page>
  );
}

function FilterStatus() {
  const status = useImageFilterStatus();
  const s: ImageFilterStatus = status.data ?? { state: "off" };

  const body = (() => {
    switch (s.state) {
      case "off":
        return {
          icon: <EyeOff />,
          title: m.images_status_off(),
          detail: m.images_status_off_detail(),
        };
      case "downloading":
        return {
          icon: <HardDriveDownload />,
          title: m.images_status_downloading(),
          detail: m.images_status_downloading_detail({ done: s.done_mb, total: s.total_mb }),
        };
      case "loading":
        return {
          icon: <Cpu />,
          title: m.images_status_loading(),
          detail: m.images_status_loading_detail(),
        };
      case "ready":
        return {
          icon: <ShieldCheck />,
          title: m.images_status_ready(),
          detail: m.images_status_ready_detail(),
        };
      case "failed":
        return {
          icon: <TriangleAlert />,
          title: m.images_status_failed(),
          detail: m.images_status_failed_detail({ error: s.error }),
        };
    }
  })();

  return (
    <Card className="mb-7" padding="md" elevation="raised">
      <div className="flex items-start gap-3">
        <span
          aria-hidden
          className={
            s.state === "failed"
              ? "text-warning [&_svg]:size-5"
              : s.state === "ready"
                ? "text-success [&_svg]:size-5"
                : "text-muted-foreground [&_svg]:size-5"
          }
        >
          {body.icon}
        </span>
        <div className="min-w-0 flex-1">
          <p className="font-medium text-foreground text-sm">{body.title}</p>
          <p className="mt-1 text-muted-foreground text-xs">{body.detail}</p>
          {s.state === "downloading" && s.total_mb > 0 && (
            <Progress
              className="mt-3"
              value={s.done_mb / s.total_mb}
              label={m.images_status_downloading()}
            />
          )}
        </div>
      </div>
    </Card>
  );
}

/**
 * One list's level. A lock may only tighten, so a locked list is offered only
 * its current level and stricter ones; the backend refuses a looser one either
 * way, and leaving it out of the menu says so before anyone tries.
 */
function ListLevel({ list, locked }: { list: BlockList; locked: boolean }) {
  const set = useSetImageFilter();
  const level = list.image_filter ?? "off";
  const current = LEVELS.indexOf(level);
  const options = LEVELS.filter((_, i) => !locked || i >= current).map((value) => ({
    value,
    ...levelLabel(value),
  }));

  return (
    <div className="px-5 py-4">
      <div className="flex items-center justify-between gap-6">
        <div className="min-w-0">
          <p className="flex items-center gap-2 font-medium text-foreground text-sm">
            <span className="truncate">{list.name}</span>
            {locked && (
              <Badge tone="warning" icon={<Lock aria-hidden />} outlined>
                {m.images_list_locked()}
              </Badge>
            )}
            {!list.enabled && <Badge tone="neutral">{m.lists_badge_off()}</Badge>}
          </p>
          <p className="mt-1 text-muted-foreground text-xs">
            {locked && level !== "off" ? m.images_list_locked_hint() : levelLabel(level).hint}
          </p>
          <InlineError error={set.error} />
        </div>
        <Select
          value={level}
          onValueChange={(next) => set.mutate({ listId: list.id, level: next })}
          options={options}
          disabled={set.isPending}
          aria-label={m.images_level_label({ name: list.name })}
          className="w-44 shrink-0"
        />
      </div>
      {level !== "off" && <SkippedSites list={list} locked={locked} />}
    </div>
  );
}

/**
 * The sites one list's filter leaves alone. A lock may only tighten, so a
 * locked list can drop a site but not add one; the add form is left out
 * rather than offered and refused.
 */
function SkippedSites({ list, locked }: { list: BlockList; locked: boolean }) {
  const save = useSetImageFilterExceptions();
  const [value, setValue] = useState("");
  const sites = list.image_filter_exceptions ?? [];

  function submit(e: FormEvent) {
    e.preventDefault();
    const site = value.trim();
    if (!site) return;
    save.mutate({ listId: list.id, sites: [...sites, site] }, { onSuccess: () => setValue("") });
  }

  return (
    <div className="mt-3 border-border border-t pt-3">
      <p className="font-medium text-foreground text-xs">{m.images_skipped_title()}</p>
      {sites.length === 0 ? (
        <p className="mt-1 text-muted-foreground text-xs">{m.images_skipped_none()}</p>
      ) : (
        <ul className="mt-2 flex flex-wrap gap-2">
          {sites.map((site) => (
            <li
              key={site}
              className="flex items-center gap-1 rounded-md border border-border bg-surface py-0.5 pr-0.5 pl-2 text-foreground text-xs"
            >
              {site}
              <button
                type="button"
                onClick={() =>
                  save.mutate({ listId: list.id, sites: sites.filter((s) => s !== site) })
                }
                disabled={save.isPending}
                aria-label={m.images_skipped_remove({ site })}
                className="rounded p-0.5 text-muted-foreground hover:bg-hover hover:text-foreground disabled:opacity-50 [&_svg]:size-3.5"
              >
                <X aria-hidden />
              </button>
            </li>
          ))}
        </ul>
      )}
      {locked ? (
        <p className="mt-2 text-muted-foreground text-xs">{m.images_skipped_locked_hint()}</p>
      ) : (
        <form onSubmit={submit} className="mt-2 flex gap-2">
          <Input
            size="sm"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="youtube.com"
            aria-label={m.images_skipped_add_label({ name: list.name })}
            className="max-w-xs"
          />
          <Button
            type="submit"
            size="sm"
            variant="outline"
            icon={<Plus />}
            disabled={!value.trim() || save.isPending}
          >
            {m.images_skipped_add()}
          </Button>
        </form>
      )}
      <InlineError error={save.error} />
    </div>
  );
}

function HowItWorks() {
  const points = [
    m.images_how_private(),
    m.images_how_extension(),
    m.images_how_download(),
    m.images_how_fails_open(),
  ];
  return (
    <section className="mb-7">
      <h2 className="font-semibold text-foreground text-sm">{m.images_how_title()}</h2>
      <Card className="mt-3" padding="md" elevation="raised">
        <ul className="list-disc space-y-2 pl-5 text-muted-foreground text-sm">
          {points.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      </Card>
    </section>
  );
}
