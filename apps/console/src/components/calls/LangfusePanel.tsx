"use client";

import { ExternalLink, Activity, CheckCircle, XCircle } from "lucide-react";

/**
 * Langfuse observability panel displayed on call detail pages.
 *
 * Shows a summary of Langfuse evaluation scores for a specific call,
 * along with a deep-link to the full trace in the Langfuse dashboard.
 */

interface LangfuseScore {
  name: string;
  value: number;
  comment?: string;
}

interface LangfusePanelProps {
  /** The QuickVoice call ID (used to build the Langfuse trace link). */
  callId: string;
  /** Base URL for the Langfuse dashboard. */
  langfuseBaseUrl?: string;
  /** Evaluation scores fetched from the call metadata. */
  scores?: LangfuseScore[];
  /** Whether Langfuse integration is enabled for this organization. */
  enabled?: boolean;
}

const SCORE_LABELS: Record<string, string> = {
  "call-duration-seconds": "Call Duration",
  "transcript-turns": "Transcript Turns",
  "data-extraction-completeness": "Data Extraction",
  "evaluation-pass-rate": "Evaluation Pass Rate",
  "call-completed": "Call Completed",
};

function formatScore(score: LangfuseScore): string {
  const name = score.name;
  if (name === "call-duration-seconds") {
    const mins = Math.floor(score.value / 60);
    const secs = Math.round(score.value % 60);
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  }
  if (
    name === "data-extraction-completeness" ||
    name === "evaluation-pass-rate"
  ) {
    return `${Math.round(score.value * 100)}%`;
  }
  if (name === "call-completed") {
    return score.value >= 1 ? "Yes" : "No";
  }
  return String(score.value);
}

function scoreColor(score: LangfuseScore): string {
  if (score.name === "call-completed") {
    return score.value >= 1
      ? "text-emerald-600 dark:text-emerald-400"
      : "text-red-500 dark:text-red-400";
  }
  if (
    score.name === "data-extraction-completeness" ||
    score.name === "evaluation-pass-rate"
  ) {
    if (score.value >= 0.8) return "text-emerald-600 dark:text-emerald-400";
    if (score.value >= 0.5) return "text-amber-500 dark:text-amber-400";
    return "text-red-500 dark:text-red-400";
  }
  return "text-foreground";
}

export function LangfusePanel({
  callId,
  langfuseBaseUrl = "http://localhost:3002",
  scores = [],
  enabled = false,
}: LangfusePanelProps) {
  if (!enabled) {
    return (
      <div className="border bg-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <Activity className="h-4 w-4 text-muted-foreground" />
          <h3 className="text-sm font-semibold">Langfuse Observability</h3>
        </div>
        <p className="text-xs text-muted-foreground">
          Langfuse evaluation is not enabled. Set{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
            LANGFUSE_ENABLED=true
          </code>{" "}
          in the AI service configuration to enable call tracing and evaluation
          scores.
        </p>
      </div>
    );
  }

  const traceUrl = `${langfuseBaseUrl.replace(/\/+$/, "")}/trace/qv-${callId}`;
  const hasScores = scores.length > 0;

  return (
    <div className="border bg-card p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-violet-500" />
          <h3 className="text-sm font-semibold">Langfuse Evaluation</h3>
        </div>
        <a
          href={traceUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          Open in Langfuse
          <ExternalLink className="h-3 w-3" />
        </a>
      </div>

      {hasScores ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {scores.map((score) => (
            <div
              key={score.name}
              className="rounded-lg border bg-muted/30 p-3 space-y-1"
            >
              <p className="text-[11px] font-medium text-muted-foreground">
                {SCORE_LABELS[score.name] ?? score.name}
              </p>
              <p className={`text-lg font-semibold ${scoreColor(score)}`}>
                {formatScore(score)}
              </p>
              {score.comment && (
                <p className="text-[10px] text-muted-foreground truncate">
                  {score.comment}
                </p>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <XCircle className="h-3.5 w-3.5" />
          <span>No evaluation scores available for this call yet.</span>
        </div>
      )}
    </div>
  );
}
