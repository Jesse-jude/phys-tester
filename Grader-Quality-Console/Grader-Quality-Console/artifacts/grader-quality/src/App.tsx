import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import {
  Activity, ArrowDownRight, ArrowRight, ArrowUpRight, Bell, Check, CheckCheck,
  ChevronDown, ChevronLeft, ChevronRight, CircleHelp, ClipboardCheck, Filter, Gauge,
  LayoutDashboard, ListFilter, Menu, MoreHorizontal, Printer, Search,
  Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Stethoscope,
  TrendingDown, Users, X,
} from 'lucide-react';
import {
  Link, Route, Switch, useLocation, Router as WouterRouter,
} from 'wouter';
import { loadConsoleData, type CaseData, type ConsoleData, type GraderData } from '@/lib/console-data';

const queryClient = new QueryClient();
const DataContext = createContext<ConsoleData | null>(null);

function useData() {
  const data = useContext(DataContext);
  if (!data) throw new Error('Console data is not loaded');
  return data;
}

const pct = (value: number | null | undefined, digits = 1) =>
  value == null ? '—' : `${(value * 100).toFixed(digits)}%`;
const percentValue = (value: number | null | undefined) =>
  value == null ? '—' : `${Math.round(value)}%`;
const barWidth = (value: number | null | undefined, scale = 100) =>
  `${Math.max(0, Math.min(100, value == null ? 0 : (value / scale) * 100))}%`;
const count = (value: number) =>
  value >= 1000 ? `${(value / 1000).toFixed(1)}k` : value.toLocaleString();
const formatDate = (value: string) =>
  new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
const initials = (value: string) => {
  const parts = value.trim().split(/[\s_]+/).filter(Boolean);
  if (parts.length <= 1) return (parts[0] ?? '').slice(0, 2).toUpperCase();
  return parts.map(part => part[0]).join('').slice(0, 2).toUpperCase();
};
const signalClass = (signal: string) =>
  signal === 'Stable' ? 'stable' : signal === 'Watch' ? 'watch' : 'review';
const reasonLabel = (reason: string) => reason.replaceAll('_', ' ');

function AppShell({ children }: { children: ReactNode }) {
  const data = useData();
  const [location] = useLocation();
  const [mobileMenu, setMobileMenu] = useState(false);
  const [notice, setNotice] = useState(false);
  const navItems = [
    { href: '/', label: 'Overview', icon: LayoutDashboard },
    { href: '/review-queue', label: 'Review queue', icon: ClipboardCheck, count: count(data.overview.casesForReview) },
    { href: '/graders', label: 'Graders', icon: Users },
    { href: '/pre-screen', label: 'Pre-screen', icon: Sparkles },
  ];
  const physicianNames = Array.from(new Set(data.graders.map(grader => grader.grader)));
  const visibleAvatars = physicianNames.slice(0, 3);
  return <div className="app-shell">
    <aside className={`sidebar ${mobileMenu ? 'sidebar-open' : ''}`} data-testid="navigation-sidebar">
      <div className="brand-lockup"><div className="brand-mark"><Activity size={19} strokeWidth={2.4} /></div><div><b>clarity</b><span>QUALITY CONSOLE</span></div><button className="icon-button sidebar-close" onClick={() => setMobileMenu(false)} aria-label="Close menu" data-testid="button-close-menu"><X size={17} /></button></div>
      <div className="workspace-switch"><span className="workspace-icon"><Stethoscope size={16} /></span><span><small>WORKSPACE</small><strong>Clinical quality</strong></span><MoreHorizontal size={14} className="muted-icon" /></div>
      <div className="nav-label">WORKSPACE</div>
      <nav className="side-nav">
        {navItems.map(({ href, label, icon: Icon, count: navCount }) => <Link key={href} href={href} onClick={() => setMobileMenu(false)} className={`nav-link ${location === href ? 'active' : ''}`} data-testid={`link-nav-${label.toLowerCase().replaceAll(' ', '-')}`}><Icon size={17} /><span>{label}</span>{navCount && <span className="nav-count">{navCount}</span>}</Link>)}
      </nav>
      <div className="side-divider" />
      <div className="nav-label">DATASET CONTEXT</div>
      <div className="dataset-context" data-testid="card-dataset-context"><div className="context-top"><span className="context-dot" /><span>{data.dataset.name}</span><MoreHorizontal size={16} /></div><p>Physician-graded benchmark</p><div className="context-stats"><div><strong>{count(data.dataset.scores)}</strong><small>scores</small></div><div><strong>{data.dataset.tasks}</strong><small>tasks</small></div><div><strong>{data.dataset.graders}</strong><small>physicians</small></div></div><div className="context-foot">Latest audit snapshot</div></div>
      <div className="sidebar-bottom"><div className="team-avatars">{visibleAvatars.map(name => <span key={name}>{initials(name)}</span>)}{physicianNames.length > visibleAvatars.length && <i>+{physicianNames.length - visibleAvatars.length}</i>}</div><div className="team-copy"><b>Quality team</b><small>{physicianNames.length} physicians</small></div><button className="icon-button" aria-label="Data settings" data-testid="button-team-settings" onClick={() => setNotice(true)}><Settings2 size={16} /></button></div>
      <div className="user-row"><div className="user-avatar">PT</div><div className="user-copy"><b>Phys-tester</b><small>Audit export</small></div><MoreHorizontal size={18} /></div>
    </aside>
    {mobileMenu && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileMenu(false)} />}
    <main className="main-area">
      <header className="topbar"><button className="icon-button mobile-menu-btn" aria-label="Open menu" onClick={() => setMobileMenu(true)} data-testid="button-open-menu"><Menu size={19} /></button><div className="breadcrumb"><span>Workspace</span><ChevronRight size={14} /><b>{navItems.find(item => item.href === location)?.label || 'Overview'}</b></div><div className="topbar-tools"><span className="sample-pill"><span />Audit data</span><span className="context-only">{formatDate(data.generatedAt)}</span><button className={`icon-button notification-button ${notice ? 'notified' : ''}`} aria-label="Notifications" onClick={() => setNotice(!notice)} data-testid="button-notifications"><Bell size={17} /><i /></button><button className="help-button" onClick={() => setNotice(!notice)} data-testid="button-help"><CircleHelp size={17} /></button></div></header>
      {notice && <div className="notice-pop" role="status"><span><Check size={15} />Local audit data</span><p>Loaded from console-data.json on {formatDate(data.generatedAt)}. Nothing is saved or sent.</p><button onClick={() => setNotice(false)} aria-label="Dismiss"><X size={14} /></button></div>}
      <div className="page-content">{children}</div>
    </main>
  </div>;
}

function PageHeading({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action && <div className="heading-action">{action}</div>}</div>;
}
function PanelTitle({ title, detail, action }: { title: string; detail?: string; action?: ReactNode }) {
  return <div className="panel-title"><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{action}</div>;
}
function Metric({ label, value, detail, positive = true, note, values }: { label: string; value: string; detail: string; positive?: boolean; note: string; values?: (number | null)[] }) {
  const heights = (values ?? []).map(value => Math.max(12, Math.min(100, (value ?? 0) * 100)));
  return <section className="metric-card" data-testid={`metric-${label.toLowerCase().replaceAll(' ', '-')}`}><div className="metric-top"><span>{label}</span><span className="metric-mark"><Activity size={15} /></span></div><div className="metric-value">{value}</div><div className={`metric-change ${positive ? 'good' : 'caution'}`}>{positive ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}<b>{detail}</b></div>{heights.length > 0 && <div className={`sparkline ${positive ? 'up' : 'down'}`}>{heights.map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}</div>}<div className="metric-note">{note}</div></section>;
}

function TaskAgreementPanel() {
  const data = useData();
  return <section className="panel agreement-panel"><PanelTitle title="Agreement by task" detail="Quadratic weighted kappa across physician pairs" action={<span className="context-only">{data.tasks.length} tasks</span>} /><div className="task-list">{data.tasks.map(task => <div className="task-row" key={task.name} data-testid={`task-agreement-${task.name}`}><div className="task-meta"><b>{task.label}</b><span>{count(task.items)} items</span></div><div className="task-bar"><span style={{ width: barWidth(task.qwk) }} /></div><div className="task-percent">{task.qwk == null ? '—' : (task.qwk * 100).toFixed(1)}<small>qwk</small></div></div>)}</div></section>;
}

function DriftPanel() {
  const data = useData();
  const signals = data.graders.filter(grader => grader.signal !== 'Stable');
  return <section className="panel drift-panel"><PanelTitle title="Drift & outliers" detail="Signals to keep an eye on" action={<Link href="/graders" className="icon-button" aria-label="Open graders" data-testid="button-drift-more"><ArrowRight size={17} /></Link>} /><div className="drift-callout"><span className="drift-icon"><TrendingDown size={17} /></span><div><b>{signals.length ? `${signals.length} grader rows need a closer look` : 'No drift signals raised'}</b><small>{signals.length ? 'Based on the latest audit' : 'All rows are within expected range'}</small></div><ArrowRight size={16} /></div>{signals.slice(0, 4).map(grader => <div className="drift-row" key={grader.id}><div className="initials-avatar">{grader.initials}</div><div className="drift-person"><b>{grader.grader}</b><small>{grader.label}</small></div><span className={`status-badge ${signalClass(grader.signal)}`}>{grader.signal}</span><span className="drift-rate">{percentValue(grader.agreementRate)}</span></div>)}{signals.length > 4 && <p className="context-foot">{signals.length - 4} more rows flagged in the grader table</p>}<Link href="/graders" className="panel-link" data-testid="link-view-graders">View all graders <ArrowRight size={14} /></Link></section>;
}

function WorkloadPanel() {
  const data = useData();
  const rate = data.overview.reviewRate ?? 0;
  return <section className="panel workload-panel"><PanelTitle title="Review workload" detail="Physician time, focused where it matters" action={<Link href="/review-queue" className="text-action" data-testid="link-workload-queue">Open queue <ArrowRight size={14}/></Link>} /><div className="workload-content"><div className="donut-wrap"><div className="donut" style={{ background: `conic-gradient(#d69c5f 0 ${rate * 100}%, #dce9df ${rate * 100}% 100%)` }}><div><b>{count(data.overview.casesForReview)}</b><small>need review</small></div></div><div className="donut-caption">cases awaiting<br/>physician review</div></div><div className="workload-stats"><div><span>Flagged grader rows</span><b>{data.overview.flaggedRows}</b></div><div><span>Insufficient data rows</span><b>{data.overview.insufficientRows}</b></div><div><span>Scores per item</span><b>{data.dataset.meanScoresPerItem?.toFixed(2) ?? '—'}</b></div><div className="workload-foot"><span>Share of all items needing review</span><strong>{pct(data.overview.reviewRate)}</strong></div></div></div></section>;
}

function ReviewPressurePanel() {
  const data = useData();
  return <section className="panel task-panel"><PanelTitle title="Review pressure by task" detail="Share of items requiring physician review" action={<Link href="/pre-screen" className="text-action" data-testid="link-prescreen-detail">Explore pre-screen <ArrowRight size={14}/></Link>} /><div className="task-list">{data.tasks.map((task, index) => <div className="task-row" key={task.name} data-testid={`task-review-${index}`}><div className="task-meta"><b>{task.label}</b><span>{count(task.needsHuman)} cases</span></div><div className="task-bar"><span style={{ width: barWidth(task.reviewRate) }} /></div><div className="task-percent">{task.reviewRate == null ? '—' : Math.round(task.reviewRate * 100)}<small>% review</small></div></div>)}</div></section>;
}

function Overview() {
  const data = useData();
  const pre = data.preScreen;
  const taskQwk = data.tasks.map(task => task.qwk);
  return <><PageHeading eyebrow="QUALITY OVERVIEW" title="Agreement & review load" description={`${data.dataset.name} · ${count(data.dataset.items)} items graded by ${data.dataset.graders} physicians across ${data.dataset.tasks} tasks.`} action={<button className="button-secondary" onClick={() => window.print()} data-testid="button-export"><Printer size={15}/>Export view</button>} />
    <div className="dataset-banner"><div className="banner-icon"><ShieldCheck size={19}/></div><div><b>{data.dataset.name}</b><span>{count(data.dataset.scores)} scores <i/> {data.dataset.tasks} tasks <i/> {data.dataset.graders} grading physicians</span></div><span className="context-only">Latest audit</span></div>
    <div className="metric-grid"><Metric label="Mean pairwise QWK" value={pct(data.overview.meanQwk)} detail={`lowest ${pct(data.overview.minQwk)}`} note={`${data.overview.totalRows} grader/task rows`} values={taskQwk} /><Metric label="Cases for review" value={count(data.overview.casesForReview)} detail={`${pct(data.overview.reviewRate)} of items`} positive={false} note="Wide spread, high IQR, or too few graders" values={data.tasks.map(task => task.reviewRate)} /><Metric label="Stable grader rows" value={`${data.overview.stableRows} / ${data.overview.totalRows}`} detail={`${data.overview.flaggedRows} flagged`} note={`${data.overview.insufficientRows} row with insufficient data`} values={data.graders.map(grader => grader.pairQwk)} /><Metric label="Judge pre-screen" value={pre.available ? pct(pre.autoRate) : 'Not run'} detail={pre.available ? `${pct(pre.autoPrecision)} precision` : 'No judge output'} positive={pre.available} note={pre.available ? `Model ${pre.model}` : 'Audit review policy only'} values={pre.available ? pre.decisions.map(decision => decision.judgeConfidence) : undefined} /></div>
    <div className="overview-main"><TaskAgreementPanel/><DriftPanel/></div>
    <div className="overview-lower"><WorkloadPanel/><ReviewPressurePanel/></div>
    <div className="prototype-foot"><span><CircleHelp size={14}/>Generated from phys-tester</span><span>{formatDate(data.generatedAt)} · audit export, not a live feed</span></div>
  </>;
}

function QueuePage() {
  const data = useData();
  const [cases, setCases] = useState<CaseData[]>(data.cases);
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('All cases');
  const [sort, setSort] = useState('Priority');
  const [saved, setSaved] = useState(false);
  const [page, setPage] = useState(1);
  const item = cases.find(entry => entry.id === selected);
  const physicians = useMemo(() => Array.from(new Set(data.graders.map(grader => grader.grader))), [data.graders]);
  const filtered = useMemo(() => cases.filter(entry => (filter === 'All cases' || (filter === 'Needs review' ? entry.status === 'Needs review' : filter === 'In review' ? entry.status === 'In review' : entry.priority === 'Urgent')) && `${entry.id} ${entry.taskLabel} ${entry.assignedTo} ${entry.flagLabels.join(' ')} ${entry.study} ${entry.model}`.toLowerCase().includes(query.toLowerCase())).sort((a, b) => sort === 'Newest' ? cases.indexOf(a) - cases.indexOf(b) : b.disagreementLevel - a.disagreementLevel), [cases, filter, query, sort]);
  const pageSize = 25;
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const currentPage = Math.min(page, pages);
  const visible = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize);
  const update = (id: string, patch: Partial<CaseData>) => setCases(current => current.map(entry => entry.id === id ? { ...entry, ...patch, updatedAt: 'just now' } : entry));
  return <><PageHeading eyebrow="PHYSICIAN WORKBENCH" title="Review queue" description="Work through the cases where physician scores diverge." action={<span className="local-note"><span/>Changes stay in this session</span>} />
    <div className="queue-summary"><div><span className="summary-number">{cases.filter(entry => entry.status !== 'Resolved').length}</span><span>open cases</span></div><div><span className="summary-number urgent-text">{cases.filter(entry => entry.priority === 'Urgent').length}</span><span>priority</span></div><div><span className="summary-number">{cases.filter(entry => entry.status === 'In review').length}</span><span>in progress</span></div><div className="summary-tail">Top cases by disagreement<br/><b>Highest first</b></div></div>
    <section className="panel queue-panel"><div className="queue-toolbar"><div className="queue-tabs">{['All cases', 'Needs review', 'In review', 'Priority'].map(option => <button key={option} className={filter === option ? 'active' : ''} onClick={() => setFilter(option)} data-testid={`filter-${option.toLowerCase().replaceAll(' ', '-')}`}>{option}{option === 'Needs review' && <i>{cases.filter(entry => entry.status === 'Needs review').length}</i>}</button>)}</div><div className="queue-tools"><label className="search-field"><Search size={16}/><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search cases…" data-testid="input-search-cases"/><kbd>⌘ K</kbd></label><button className="button-secondary compact" onClick={() => setSort(sort === 'Priority' ? 'Newest' : 'Priority')} data-testid="button-sort"><SlidersHorizontal size={15}/>{sort}<ChevronDown size={14}/></button></div></div>
      <div className="queue-table"><div className="table-head"><span>CASE / TASK</span><span>PRIORITY</span><span>AGREEMENT SIGNAL</span><span>ASSIGNED TO</span><span>STATUS</span><span>FLAGS</span></div>{visible.map(entry => <button className={`case-row ${selected === entry.id ? 'selected' : ''}`} key={entry.id} onClick={() => { setSelected(entry.id); setSaved(false); }} data-testid={`row-case-${entry.id}`}><div className="case-cell"><span className="case-id">#{entry.id}</span><b>{entry.taskLabel}</b></div><span><i className={`priority-dot ${entry.priority.toLowerCase()}`}/>{entry.priority}</span><div className="agreement-cell"><b>{Math.round(entry.disagreementLevel)}%</b><span className="agreement-track"><i style={{ width: barWidth(entry.disagreementLevel) }}/></span></div><span className="assigned-cell"><span className="tiny-avatar">{entry.assignedTo === 'Unassigned' ? '+' : initials(entry.assignedTo)}</span>{entry.assignedTo}</span><span><i className={`status-point ${entry.status === 'Resolved' ? 'resolved' : entry.status === 'In review' ? 'inreview' : ''}`}/>{entry.status}</span><span className="updated-cell" title={entry.flagLabels.join(', ')}>{entry.flagLabels.length ? `${entry.flagLabels[0]}${entry.flagLabels.length > 1 ? ` +${entry.flagLabels.length - 1}` : ''}` : '—'}</span></button>)}{filtered.length === 0 && <div className="empty-queue"><ListFilter size={22}/><b>No cases match these filters</b><span>Try a different search or queue view.</span><button onClick={() => { setQuery(''); setFilter('All cases'); }} data-testid="button-clear-filters">Clear filters</button></div>}</div>
      <div className="table-footer"><span>Showing <b>{visible.length}</b> of {filtered.length} matching · {count(data.overview.casesForReview)} need review</span><span><button className="icon-button" disabled={currentPage === 1} onClick={() => setPage(currentPage - 1)} aria-label="Previous page" style={{ opacity: currentPage === 1 ? 0.35 : 1 }}><ChevronLeft size={15}/></button>Page <b>{currentPage}</b> of {pages}<button className="icon-button" disabled={currentPage === pages} onClick={() => setPage(currentPage + 1)} aria-label="Next page" style={{ opacity: currentPage === pages ? 0.35 : 1 }}><ChevronRight size={15}/></button></span></div>
    </section>
    {item && <div className="detail-scrim" onClick={() => setSelected(null)}><aside className="case-detail" onClick={event => event.stopPropagation()}><div className="detail-header"><div><span className="eyebrow">AUDIT CASE</span><h2>#{item.id}</h2></div><button className="icon-button" onClick={() => setSelected(null)} aria-label="Close case details" data-testid="button-close-case"><X size={18}/></button></div><div className="detail-task"><div className="detail-task-icon"><Activity size={18}/></div><div><b>{item.taskLabel}</b><span>{item.task} · scale 0–{Math.round(item.scoreScale)} · {item.study}</span></div></div><div className="detail-section"><div className="detail-label">PHYSICIAN COMPARISON <span>Scale 0–{Math.round(item.scoreScale)}</span></div>{item.scores.map((score, index) => <div className="score-row" key={`${score.name}-${index}`}><div className={`score-avatar ${['sage', 'coral', 'blue'][index % 3]}`}>{initials(score.name)}</div><span>{score.name}</span><div className="score-track">{Array.from({ length: Math.max(1, Math.round(item.scoreScale)) }, (_, segment) => segment + 1).map(step => <i key={step} className={step <= Math.round(score.score) ? 'filled' : ''}/>)}</div><b>{score.score}</b></div>)}<div className="range-note"><span>Observed spread</span><b>{pct(item.spreadFraction)} · {item.flagLabels.join(' · ') || 'no flags'}</b></div></div><div className="detail-section"><div className="detail-label">RESPONSE EXCERPT <span>{item.model}</span></div><p style={{ margin: 0, fontSize: 11, lineHeight: 1.6, color: '#5e706a', whiteSpace: 'pre-wrap' }}>{item.response || 'No response text exported.'}</p>{item.reference && <p style={{ margin: '10px 0 0', fontSize: 10, lineHeight: 1.5, color: '#85938d' }}>Reference: {item.reference}</p>}</div>{item.llmConfidence == null ? <div className="confidence-box"><div><Sparkles size={15}/><b>Judge pre-screen not run</b></div><span>n/a</span><small>No judge output exists for this dataset. This case is queued by review policy.</small></div> : <div className="confidence-box"><div><Sparkles size={15}/><b>Judge pre-screen confidence</b></div><span>{Math.round(item.llmConfidence * 100)}%</span><div className="confidence-bar"><i style={{ width: barWidth(item.llmConfidence, 1) }}/></div><small>Model confidence from the routing export.</small></div>}<div className="detail-section assignment"><div className="detail-label">ASSIGN PHYSICIAN</div><select value={item.assignedTo} onChange={event => update(item.id, { assignedTo: event.target.value, status: event.target.value === 'Unassigned' ? 'Needs review' : 'In review' })} data-testid="select-assignee"><option>Unassigned</option>{physicians.map(name => <option key={name}>{name}</option>)}</select></div><div className="detail-actions">{item.status !== 'Resolved' ? <button className="button-primary full-button" onClick={() => { update(item.id, { status: 'Resolved' }); setSaved(true); }} data-testid="button-resolve-case"><CheckCheck size={16}/>Mark resolved</button> : <button className="button-secondary full-button" onClick={() => { update(item.id, { status: 'Needs review' }); setSaved(false); }} data-testid="button-reopen-case"><Activity size={16}/>Reopen case</button>}<button className="button-secondary full-button" onClick={() => { setSaved(true); update(item.id, { status: 'In review' }); }} data-testid="button-save-review"><Check size={16}/>{saved ? 'Saved in this session' : 'Save review note'}</button>{saved && <span className="save-note"><Check size={13}/> Local session state updated</span>}</div><div className="detail-foot">Source: {item.study} · {item.model} · latest audit</div></aside></div>}
  </>;
}

function GradersPage() {
  const data = useData();
  const [taskFilter, setTaskFilter] = useState('All tasks');
  const [search, setSearch] = useState('');
  const [selectedGrader, setSelectedGrader] = useState<string | null>(null);
  const list = data.graders.filter(grader => (taskFilter === 'All tasks' || grader.label === taskFilter) && `${grader.grader} ${grader.label} ${grader.id}`.toLowerCase().includes(search.toLowerCase()));
  const selected: GraderData | undefined = data.graders.find(grader => grader.id === selectedGrader);
  const signalCounts = {
    stable: data.graders.filter(grader => grader.signal === 'Stable').length,
    watch: data.graders.filter(grader => grader.signal === 'Watch').length,
    review: data.graders.filter(grader => grader.signal !== 'Stable' && grader.signal !== 'Watch').length,
  };
  return <><PageHeading eyebrow="GRADER MONITORING" title="Grader quality" description="Agreement, bias, and drift signals for each physician and task." action={<span className="context-only">{data.overview.totalRows} rows</span>} />
    <div className="grader-summary"><div className="summary-feature"><div className="summary-feature-icon"><Gauge size={18}/></div><div><span>Mean pairwise QWK</span><b>{pct(data.overview.meanQwk)}</b><small>Across {data.tasks.length} tasks</small></div><div className="summary-mini"><div className="mini-bars">{data.tasks.map(task => <div className="bar-column" key={task.name}><div className="bar-track"><span style={{ height: `${Math.max(8, (task.qwk ?? 0) * 100)}%` }} /></div><small>{task.name.slice(0, 3)}</small></div>)}</div></div></div><div className="grader-summary-stat"><span className="status-badge stable">Stable</span><b>{signalCounts.stable}</b><small>within expected range</small></div><div className="grader-summary-stat"><span className="status-badge watch">Watch</span><b>{signalCounts.watch}</b><small>mild signal observed</small></div><div className="grader-summary-stat"><span className="status-badge review">Review</span><b>{signalCounts.review}</b><small>review or insufficient data</small></div></div>
    <section className="panel graders-panel"><PanelTitle title="Physician graders" detail="Each row is a physician/task pair; bias flags are shown only when estimable." action={<button className="button-secondary compact" onClick={() => { setTaskFilter('All tasks'); setSearch(''); }} data-testid="button-reset-grader-filters"><Filter size={15}/>Reset</button>} /><div className="grader-filters"><label className="search-field"><Search size={15}/><input placeholder="Find a grader…" value={search} onChange={event => setSearch(event.target.value)} data-testid="input-search-graders"/></label><select value={taskFilter} onChange={event => setTaskFilter(event.target.value)} data-testid="select-task-filter"><option>All tasks</option>{data.tasks.map(task => <option key={task.name}>{task.label}</option>)}</select></div><div className="grader-table"><div className="grader-head"><span>PHYSICIAN</span><span>SCORES</span><span>PAIR QWK</span><span>TASK</span><span>SIGNAL</span><span></span></div>{list.map(grader => <button className="grader-row" onClick={() => setSelectedGrader(grader.id)} key={grader.id} data-testid={`row-grader-${grader.id}`}><div className="grader-name"><span className="initials-avatar">{grader.initials}</span><span><b>{grader.grader}</b><small>{grader.specialty}</small></span></div><b>{count(grader.scores)}</b><div className="grader-agreement"><b>{percentValue(grader.agreementRate)}</b><span className="mini-progress"><i style={{ width: barWidth(grader.agreementRate) }}/></span></div><span>{grader.label}</span><span><i className={`status-point ${grader.signal === 'Stable' ? 'resolved' : grader.signal === 'Watch' ? 'inreview' : ''}`}/><span className={`status-badge ${signalClass(grader.signal)}`}>{grader.signal === 'Insufficient data' ? 'Insufficient' : grader.signal}</span></span><ChevronRight size={16}/></button>)}{!list.length && <div className="empty-queue"><Users size={22}/><b>No graders found</b><span>Adjust your task or search filters.</span></div>}</div><div className="table-footer"><span>{list.length} of {data.graders.length} physician/task rows</span><span>{data.overview.flaggedRows} flagged · {data.overview.insufficientRows} insufficient</span></div></section>
    <section className="panel task-context-panel"><PanelTitle title="Task-level context" detail="Agreement and review load by grading task" action={<span className="context-only">Latest audit</span>} /><div className="task-context-grid">{data.tasks.map(task => <div className="task-context-card" key={task.name}><div className="task-context-top"><span>{task.label}</span><MoreHorizontal size={16}/></div><b>{task.qwk == null ? '—' : (task.qwk * 100).toFixed(1)}<small>%</small></b><div className="task-context-foot"><span>{task.graders} graders</span><span>{count(task.items)} items</span></div><div className="task-context-bar"><i style={{ width: barWidth(task.qwk) }}/></div></div>)}</div></section>
    {selected && <div className="detail-scrim" onClick={() => setSelectedGrader(null)}><aside className="case-detail grader-detail" onClick={event => event.stopPropagation()}><div className="detail-header"><div><span className="eyebrow">GRADER/TASK PROFILE</span><h2>{selected.grader}</h2><p>{selected.label} · {selected.id}</p></div><button className="icon-button" onClick={() => setSelectedGrader(null)} aria-label="Close grader profile" data-testid="button-close-grader"><X size={18}/></button></div><div className="grader-detail-score"><b>{percentValue(selected.agreementRate)}</b><span>pairwise agreement rate</span><div className="task-context-bar"><i style={{ width: barWidth(selected.agreementRate) }}/></div></div><div className="detail-section"><div className="detail-label">MONITORING CONTEXT</div><div className="context-detail-row"><span>Scores</span><b>{count(selected.scores)}</b></div><div className="context-detail-row"><span>Coverage</span><b>{pct(selected.coverage)}</b></div><div className="context-detail-row"><span>Pair count</span><b>{selected.pairCount}</b></div><div className="context-detail-row"><span>Pairwise QWK</span><b>{selected.pairQwk == null ? '—' : selected.pairQwk.toFixed(3)}</b></div><div className="context-detail-row"><span>Bias z</span><b>{selected.biasEstimable && selected.biasZ != null ? selected.biasZ.toFixed(2) : 'not estimable'}</b></div><div className="context-detail-row"><span>Bias effect</span><b>{pct(selected.biasEffect)}</b></div><div className="context-detail-row"><span>Component size</span><b>{selected.componentSize}</b></div><div className="context-detail-row"><span>Outlier rate</span><b>{pct(selected.outlierRate)}</b></div><div className="context-detail-row"><span>Drift rho</span><b>{selected.driftRho == null ? '—' : selected.driftRho.toFixed(2)}</b></div><div className="context-detail-row"><span>Flags</span><b>{selected.flags.map(reasonLabel).join(', ') || 'none'}</b></div></div><div className="confidence-box"><div><Activity size={15}/><b>Signal</b></div><span>{selected.signal}</span><small>Bias flags require at least two partners and a minimum effect size.</small></div><button className="button-primary full-button" onClick={() => setSelectedGrader(null)} data-testid="button-close-profile">Done</button></aside></div>}
  </>;
}

type RouteRow = {
  id: string;
  taskLabel: string;
  priority: string;
  confidence: number | null;
  route: string;
  reasons: string[];
};

function PreScreenPage() {
  const data = useData();
  const pre = data.preScreen;
  const [tab, setTab] = useState('All');
  const auditAutoRate = data.overview.reviewRate == null ? null : 1 - data.overview.reviewRate;
  const rows: RouteRow[] = pre.available
    ? pre.decisions.map(decision => ({ id: decision.id, taskLabel: decision.taskLabel, priority: decision.route.includes('auto') ? 'Routine' : 'Urgent', confidence: decision.judgeConfidence, route: decision.route.includes('auto') ? 'Auto-routed' : 'Physician review', reasons: decision.reasons }))
    : data.cases.map(item => ({ id: item.id, taskLabel: item.taskLabel, priority: item.priority, confidence: item.spreadFraction, route: 'Physician review', reasons: item.flagLabels }));
  const tabs = pre.available ? ['All', 'Physician review', 'Auto-routed'] : ['All', 'Urgent', 'Wide spread'];
  const isAuto = (row: RouteRow) => row.route.toLowerCase().includes('auto');
  const matches = (row: RouteRow) => tab === 'All' ? true : tab === 'Urgent' ? row.priority === 'Urgent' : tab === 'Wide spread' ? row.reasons.includes('wide spread') : tab === 'Auto-routed' ? isAuto(row) : !isAuto(row);
  const filtered = rows.filter(matches);
  const tabCount = (option: string) => option === 'All' ? rows.length : rows.filter(row => option === 'Urgent' ? row.priority === 'Urgent' : option === 'Wide spread' ? row.reasons.includes('wide spread') : option === 'Auto-routed' ? isAuto(row) : !isAuto(row)).length;
  return <><PageHeading eyebrow="JUDGE PRE-SCREEN" title="Pre-screen routing" description={pre.available ? 'Judge confidence and routing results from the latest export.' : 'Judge pre-screen has not been run for this dataset. This page reflects the audit review policy only.'} action={<span className="demo-badge"><Sparkles size={14}/>{pre.available ? pre.model : 'Judge not run'}</span>} />
    <div className="prescreen-warning"><div><ShieldCheck size={17}/><b>{pre.available ? 'Judge results are advisory' : 'No judge output in this export'}</b></div><span>{pre.available ? 'Physician scores remain the reference standard. Routing decisions do not replace review.' : 'Run phys-tester route and rebuild console-data.json with --routing to populate judge decisions and calibration.'}</span></div>
    <div className="prescreen-stats"><div className="prescreen-stat auto-stat"><span className="prescreen-icon"><ArrowRight size={17}/></span><span>Audit-policy auto share</span><b>{pct(auditAutoRate)}</b><small>{count(Math.max(data.dataset.scoredItems - data.overview.casesForReview, 0))} scored items outside review</small></div><div className="prescreen-stat escalated-stat"><span className="prescreen-icon"><Users size={17}/></span><span>Sent to physicians</span><b>{count(data.overview.casesForReview)}</b><small>{pct(data.overview.reviewRate)} of all items</small></div><div className="prescreen-stat"><span className="prescreen-icon"><Gauge size={17}/></span><span>Judge pre-screen</span><b>{pre.available ? pct(pre.autoRate) : 'Not run'}</b><small>{pre.available ? `${pct(pre.autoPrecision)} auto precision` : 'No model output'}</small></div><div className="prescreen-stat model-stat"><span className="prescreen-icon"><Sparkles size={17}/></span><span>Model label</span><b>{pre.model ?? '—'}</b><small>{pre.available ? `Threshold ${pre.threshold}` : 'Populated by routing export'}</small></div></div>
    <section className="panel routing-panel"><PanelTitle title={pre.available ? 'Routing threshold' : 'Audit review policy'} detail={pre.available ? 'Judge confidence split from the latest export' : 'Rules that place an item in the review queue'} action={<span className="read-only-label">{pre.available ? 'FROM ROUTING EXPORT' : 'NO JUDGE RUN'}</span>} /><div className="routing-numbers">{pre.available ? <><div><span>Threshold</span><b>≥ {pre.threshold}</b></div><div><span>Auto rate</span><b>{pct(pre.autoRate)}</b></div><div><span>Auto precision</span><b>{pct(pre.autoPrecision)}</b></div><div><span>Hard-case recall</span><b>{pct(pre.hardCaseRecall)}</b></div></> : <><div><span>Minimum graders</span><b>{data.reviewConfig.min_graders ?? '—'}</b></div><div><span>Wide spread</span><b>≥ {pct(data.reviewConfig.spread_fraction)}</b></div><div><span>High IQR</span><b>≥ {pct(data.reviewConfig.iqr_fraction)}</b></div><div><span>Bimodal gap</span><b>≥ {pct(data.reviewConfig.bimodal_gap_fraction)} · {data.reviewConfig.bimodal_min_group ?? '—'} per group</b></div></>}</div><div className="routing-note"><CircleHelp size={15}/>{pre.available ? 'Judge routing is evaluated against physician consensus; see the audit report for calibration details.' : 'These thresholds come from the audit configuration. No confidence threshold is simulated here.'}</div></section>
    <section className="panel prescreen-cases"><PanelTitle title={pre.available ? 'Routing decisions' : 'Review-policy examples'} detail={pre.available ? 'Judge decisions from the latest export' : 'Top disagreement cases from the audit'} action={<Link href="/review-queue" className="button-secondary compact">Open queue</Link>} /><div className="prescreen-tabs">{tabs.map(option => <button key={option} className={tab === option ? 'active' : ''} onClick={() => setTab(option)} data-testid={`prescreen-tab-${option.toLowerCase().replaceAll(' ', '-')}`}>{option}{option !== 'All' && <i>{tabCount(option)}</i>}</button>)}</div><div className="prescreen-table"><div className="prescreen-head"><span>EXAMPLE</span><span>TASK</span><span>{pre.available ? 'CONFIDENCE' : 'SPREAD'}</span><span>ROUTING</span><span>RATIONALE</span><span></span></div>{filtered.slice(0, 10).map(row => <div className="prescreen-row" key={row.id} data-testid={`prescreen-case-${row.id}`}><span className="case-id">#{row.id}</span><b>{row.taskLabel}</b><div className="confidence-value"><b>{row.confidence == null ? '—' : `${Math.round(row.confidence * 100)}%`}</b><span><i style={{ width: barWidth(row.confidence, 1) }}/></span></div><span className={`route-tag ${row.route.toLowerCase().includes('auto') ? 'route-auto' : 'route-review'}`}><i/>{row.route}</span><span className="route-reason">{row.reasons.map(reasonLabel).join(', ') || '—'}</span><span /></div>)}{!filtered.length && <div className="empty-queue">No cases in this view.</div>}</div><div className="prescreen-footer"><span>{pre.available ? `${pre.decisions.length} judge decisions exported` : `${data.cases.length} top review cases shown`}</span><span><span className="route-legend-dot"/>Automatically routed <span className="route-legend-dot review-dot"/>Physician review</span></div></section>
    <div className="prototype-foot"><span><Sparkles size={14}/>{pre.available ? 'Judge pre-screen export' : 'Audit-policy fallback'}</span><span>{pre.available ? `Model ${pre.model} · advisory only` : 'No judge calls are made by this console.'}</span></div>
  </>;
}

function Router() {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}><AppShell><Switch><Route path="/" component={Overview}/><Route path="/review-queue" component={QueuePage}/><Route path="/graders" component={GradersPage}/><Route path="/pre-screen" component={PreScreenPage}/><Route component={NotFound}/></Switch></AppShell></ErrorBoundary>;
}

function LoadingScreen({ error }: { error: string | null }) {
  return <div style={{ minHeight: '100dvh', display: 'grid', placeItems: 'center', background: '#faf9f5', padding: 24, fontFamily: 'Inter, system-ui, sans-serif' }}>
    <div style={{ maxWidth: 460, textAlign: 'center', color: '#526563' }}>
      <div style={{ width: 42, height: 42, borderRadius: 12, background: '#eaf1e9', color: '#3f846b', display: 'grid', placeItems: 'center', margin: '0 auto 14px' }}><Activity size={22} /></div>
      <h1 style={{ fontSize: 18, margin: '0 0 8px', color: '#33484a' }}>{error ? 'Console data could not be loaded' : 'Loading audit data'}</h1>
      <p style={{ fontSize: 12, lineHeight: 1.6, margin: 0 }}>{error ?? 'Reading console-data.json from the public directory.'}</p>
      {error && <p style={{ fontSize: 11, lineHeight: 1.6, margin: '12px 0 0', color: '#8a6a52' }}>Generate it with <code>phys-tester console</code> and reload.</p>}
    </div>
  </div>;
}

function App() {
  const [data, setData] = useState<ConsoleData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    loadConsoleData().then(setData).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Unknown error'));
  }, []);
  if (!data) return <LoadingScreen error={error} />;
  return <DataContext.Provider value={data}><QueryClientProvider client={queryClient}><TooltipProvider><WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><Router/></WouterRouter><Toaster/></TooltipProvider></QueryClientProvider></DataContext.Provider>;
}
export default App;
