# Grader Quality Console

Static React console that renders a Phys-tester physician-grader quality audit. It reads `console-data.json` from `public/` and does not call any backend or model.

## Run & Operate

- `pnpm --filter @workspace/grader-quality run dev` — run the console (needs `PORT` and `BASE_PATH`)
- `pnpm --filter @workspace/grader-quality run build` — build the static bundle
- `pnpm --filter @workspace/grader-quality run typecheck` — typecheck the console package
- `pnpm run typecheck` / `pnpm run build` — workspace-wide checks

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9, Vite
- React 19 + wouter routing, Tailwind v4, Radix UI primitives
- Data: static JSON fetched from `public/console-data.json`

## Where things live

- `artifacts/grader-quality/src/App.tsx` — every page (Overview, Review queue, Graders, Pre-screen)
- `artifacts/grader-quality/src/lib/console-data.ts` — payload types and loader
- `artifacts/grader-quality/public/console-data.json` — audit data, generated, do not hand-edit
- `artifacts/grader-quality/src/index.css` — theme and component classes

## Architecture decisions

- The console is view-only. It never scores, routes, or persists; session-only case actions are local state.
- All metrics come from the audit export. Nothing is synthesized, so any "not run" state is shown honestly instead of filled with sample values.
- Judge pre-screen is optional. Without a routing export the page shows the audit review policy and states that no judge was run.
- `console-data.json` is committed into `public/` so the static build has no runtime dependency on the Python tool.

## Product

- Overview: mean pairwise QWK, review load, stable/flagged grader rows, agreement and review pressure by task.
- Review queue: highest-disagreement cases with physician scores, flags, response excerpts, and session-only assignment/resolution.
- Graders: physician/task agreement, bias diagnostics (only when estimable), drift signals, and per-task context.
- Pre-screen: judge routing results when a routing export is supplied, otherwise the audit review policy and its thresholds.

## User preferences

- Use real audit data only. Do not introduce sample/demo datasets or fake judge confidences.
- Keep the existing visual design and CSS classes; make the pages data-driven.

## Gotchas

- Regenerate `console-data.json` with `phys-tester console` after any data or config change.
- Pass `--routing reports/routing.json` only when a real judge run exists; otherwise the pre-screen page reports "judge not run".
- The workspace references `lib/*` and `scripts` that are not checked in, so `pnpm install` and workspace typecheck can fail even though the console does not use them.
- `vite.config.ts` throws without `PORT` and `BASE_PATH`; `BASE_PATH` must end with `/` for the data fetch to resolve.
- To run locally without pnpm, use the npm wrapper one directory above this repo: `cd .. && npm install && npm run dev` (http://127.0.0.1:5173).

## Pointers

- Generate data: `phys-tester console --data <dataset> --audit <audit.json> --out artifacts/grader-quality/public/console-data.json`
- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
