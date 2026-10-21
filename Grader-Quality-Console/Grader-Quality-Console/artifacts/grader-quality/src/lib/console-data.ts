export type TaskData = {
  name: string;
  label: string;
  items: number;
  scoredItems: number;
  scores: number;
  graders: number;
  qwk: number | null;
  icc: number | null;
  alpha: number | null;
  exact: number | null;
  spread: number | null;
  needsHuman: number;
  reviewRate: number | null;
  flagCounts: Record<string, number>;
  scoreScale: string;
  normalized: boolean;
};

export type GraderData = {
  id: string;
  grader: string;
  task: string;
  label: string;
  initials: string;
  specialty: string;
  scores: number;
  coverage: number | null;
  agreementRate: number | null;
  pairQwk: number | null;
  pairCount: number;
  biasZ: number | null;
  biasEstimable: boolean;
  biasEffect: number | null;
  componentSize: number;
  outlierRate: number | null;
  driftRho: number | null;
  flags: string[];
  status: string;
  signal: string;
};

export type CaseScore = {
  name: string;
  score: number;
};

export type CaseData = {
  id: string;
  task: string;
  taskLabel: string;
  priority: 'Urgent' | 'Review' | 'Routine';
  disagreementLevel: number;
  spreadFraction: number | null;
  flags: string[];
  flagLabels: string[];
  nScores: number;
  scoreScale: number;
  scores: CaseScore[];
  response: string;
  reference: string;
  study: string;
  model: string;
  status: 'Needs review' | 'In review' | 'Resolved';
  assignedTo: string;
  updatedAt: string;
  llmConfidence: number | null;
};

export type PreScreenDecision = {
  id: string;
  task: string;
  taskLabel: string;
  route: string;
  reasons: string[];
  judgeScore: number | null;
  judgeConfidence: number | null;
  physicianSpread: number | null;
  withinTolerance: boolean | null;
  correct: boolean | null;
};

export type PreScreenData = {
  available: boolean;
  model: string | null;
  threshold: number | null;
  autoRate: number | null;
  autoPrecision: number | null;
  hardCaseRecall: number | null;
  reasonCounts: Record<string, number>;
  byTask: Record<string, unknown>;
  decisions: PreScreenDecision[];
};

export type ConsoleData = {
  version: number;
  generatedAt: string;
  dataset: {
    name: string;
    items: number;
    scoredItems: number;
    scores: number;
    tasks: number;
    graders: number;
    models: number;
    studies: number;
    meanScoresPerItem: number | null;
  };
  overview: {
    meanQwk: number | null;
    minQwk: number | null;
    casesForReview: number;
    reviewRate: number | null;
    stableRows: number;
    flaggedRows: number;
    insufficientRows: number;
    totalRows: number;
    uniqueGraders: number;
    autoCoverage: number | null;
    autoPrecision: number | null;
    judgeModel: string | null;
  };
  tasks: TaskData[];
  graders: GraderData[];
  cases: CaseData[];
  preScreen: PreScreenData;
  reviewConfig: {
    min_graders?: number;
    spread_fraction?: number;
    iqr_fraction?: number;
    bimodal_gap_fraction?: number;
    bimodal_min_group?: number;
  };
};

export async function loadConsoleData(): Promise<ConsoleData> {
  const response = await fetch(`${import.meta.env.BASE_URL}console-data.json`, {
    cache: 'no-store',
  });
  if (!response.ok) {
    throw new Error(`Could not load console-data.json (${response.status})`);
  }
  const data = (await response.json()) as ConsoleData;
  if (!data || data.version !== 1) {
    throw new Error('Unsupported console-data.json version');
  }
  return data;
}
