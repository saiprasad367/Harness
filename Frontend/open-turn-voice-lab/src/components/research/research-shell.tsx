import { useState, type ReactNode } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import { AudioLines, Bell, ChevronDown, CircleHelp, Command, Headphones, Menu, Mic2, PanelLeftClose, PanelLeftOpen, Search, Signal, Sparkles, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import { navigation, recordings, type ResearchPage } from "@/lib/research-data";
import { toast } from "sonner";

const pageHeadings: Record<ResearchPage, string> = {
  overview: "Overview", voice: "Live Voice Studio", simulation: "Simulation Studio", datasets: "Dataset Explorer",
  annotation: "Ground-Truth Annotation", evaluation: "Evaluation & Benchmarks", reports: "Research Reports", settings: "Settings",
};

export function ResearchShell({ page, children }: { page: ResearchPage; children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [noticesOpen, setNoticesOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const pathname = useRouterState({ select: (router) => router.location.pathname });
  const matchingRecords = search.trim() ? recordings.filter((record) => `${record.id} ${record.speaker} ${record.cohort}`.toLowerCase().includes(search.toLowerCase())).slice(0, 4) : [];
  const matchingPages = search.trim() ? navigation.filter((item) => item.label.toLowerCase().includes(search.toLowerCase())).slice(0, 3) : [];

  return (
    <TooltipProvider delayDuration={250}>
      <div className="flex min-h-screen bg-background text-foreground">
        <aside className={`fixed inset-y-0 left-0 z-40 flex flex-col border-r border-sidebar-border bg-sidebar transition-all duration-200 md:static md:translate-x-0 ${mobileOpen ? "translate-x-0" : "-translate-x-full"} ${collapsed ? "w-[76px]" : "w-[264px]"}`}>
          <div className={`flex h-[72px] items-center border-b border-sidebar-border ${collapsed ? "justify-center px-3" : "gap-3 px-5"}`}>
            <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm"><AudioLines className="size-5" /></div>
            {!collapsed && <div className="min-w-0"><p className="font-display text-sm font-extrabold leading-5">Duplex Lab</p><p className="text-[11px] text-muted-foreground">Voice research workspace</p></div>}
            <Button aria-label="Close navigation" variant="ghost" size="icon" className="ml-auto md:hidden" onClick={() => setMobileOpen(false)}><X /></Button>
          </div>
          <div className={`px-3 pt-5 ${collapsed ? "text-center" : ""}`}>
            {!collapsed && <p className="px-2 pb-2 text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">Research</p>}
            <nav aria-label="Main navigation" className="space-y-1">
              {navigation.map((item) => {
                const Icon = item.icon;
                const active = pathname === item.path;
                const link = <Link to={item.path} onClick={() => setMobileOpen(false)} aria-current={active ? "page" : undefined} className={`flex h-10 items-center gap-3 rounded-lg px-3 text-[13px] font-medium transition-colors ${active ? "bg-secondary text-secondary-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`}>
                  <Icon className={`size-[17px] shrink-0 ${active ? "text-primary" : ""}`} />{!collapsed && <span className="truncate">{item.label}</span>}{active && !collapsed && <span className="ml-auto size-1.5 rounded-full bg-primary" />}
                </Link>;
                return <div key={item.key}>{collapsed ? <Tooltip><TooltipTrigger asChild>{link}</TooltipTrigger><TooltipContent side="right">{item.label}</TooltipContent></Tooltip> : link}</div>;
              })}
            </nav>
          </div>
          <div className="mt-auto px-3 pb-3">
            {!collapsed && <div className="mb-3 rounded-xl border border-border bg-background p-3">
              <div className="mb-2 flex items-center justify-between"><span className="text-xs font-semibold">Workspace health</span><span className="flex items-center gap-1.5 text-[10px] font-medium text-emerald-700"><span className="size-1.5 rounded-full bg-emerald-500" />Active</span></div>
              <div className="h-1.5 overflow-hidden rounded-full bg-muted"><div className="h-full w-[100%] rounded-full bg-primary" /></div>
              <p className="mt-2 text-[10px] text-muted-foreground">Full-Duplex Pipeline · Genuine Moshi GPU</p>
            </div>}
            <Button aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} variant="ghost" className={`w-full text-muted-foreground ${collapsed ? "justify-center px-0" : "justify-start"}`} onClick={() => setCollapsed(!collapsed)}>
              {collapsed ? <PanelLeftOpen /> : <><PanelLeftClose /><span>Collapse sidebar</span></>}
            </Button>
          </div>
        </aside>
        {mobileOpen && <button aria-label="Close menu overlay" className="fixed inset-0 z-30 bg-foreground/20 md:hidden" onClick={() => setMobileOpen(false)} />}

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-20 flex h-[72px] items-center justify-between gap-4 border-b border-border bg-card/95 px-4 backdrop-blur md:px-7">
            <div className="flex min-w-0 items-center gap-3">
              <Button aria-label="Open navigation" variant="ghost" size="icon" className="md:hidden" onClick={() => setMobileOpen(true)}><Menu /></Button>
              <div className="hidden min-w-0 sm:block"><p className="truncate text-sm font-semibold">{pageHeadings[page]}</p><p className="text-[11px] text-muted-foreground">Research workspace <span className="px-1">/</span> {pageHeadings[page]}</p></div>
            </div>
            <div className="flex min-w-0 items-center gap-2 sm:gap-3">
              <div className="relative hidden w-[190px] lg:block xl:w-[250px]">
                <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input aria-label="Search workspace" placeholder="Search workspace…" value={search} onChange={(event) => setSearch(event.target.value)} className="h-9 border-0 bg-muted pl-9 pr-12 text-xs shadow-none" onKeyDown={(event) => { if (event.key === "Escape") setSearch(""); }} />
                <span className="absolute right-2 top-1/2 flex -translate-y-1/2 items-center gap-0.5 text-[10px] text-muted-foreground"><Command className="size-3" />K</span>
                {search && <div className="absolute right-0 top-11 z-50 w-[290px] rounded-xl border border-border bg-popover p-2 shadow-lg">
                  {matchingPages.length + matchingRecords.length === 0 ? <p className="p-3 text-xs text-muted-foreground">No matching pages or recordings.</p> : <>
                    {matchingPages.map((item) => <Link key={item.key} to={item.path} onClick={() => setSearch("")} className="block rounded-lg px-3 py-2 text-xs hover:bg-muted">Page · {item.label}</Link>)}
                    {matchingRecords.map((record) => <Link key={record.id} to="/datasets" onClick={() => setSearch("")} className="block rounded-lg px-3 py-2 text-xs hover:bg-muted">Recording · {record.id} <span className="text-muted-foreground">{record.speaker}</span></Link>)}
                  </>}
                </div>}
              </div>
              <div className="hidden items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1.5 sm:flex"><Signal className="size-3.5 text-emerald-600 animate-pulse" /><span className="text-[10px] font-semibold text-emerald-700">Moshi GPU Active</span></div>
              <Button aria-label="Help and documentation" title="Help and documentation" variant="ghost" size="icon" onClick={() => setHelpOpen(!helpOpen)}><CircleHelp /></Button>
              <div className="relative">
                <Button aria-label="Notifications" title="Notifications" variant="ghost" size="icon" onClick={() => setNoticesOpen(!noticesOpen)}><Bell /><span className="absolute right-2 top-2 size-1.5 rounded-full bg-primary" /></Button>
                {noticesOpen && <div className="absolute right-0 top-11 z-50 w-72 rounded-xl border border-border bg-popover p-4 shadow-lg"><div className="mb-3 flex items-center justify-between"><span className="text-sm font-semibold">Notifications</span><Badge variant="secondary">2 new</Badge></div><p className="border-b border-border py-2 text-xs">Evaluation preview is ready <span className="block pt-1 text-muted-foreground">EXP-2025-041 · mock result</span></p><p className="pt-2 text-xs">3 candidate pauses need review <span className="block pt-1 text-muted-foreground">Annotation queue</span></p></div>}
              </div>
              <div className="hidden h-7 w-px bg-border sm:block" />
              <div className="flex items-center gap-2"><div className="flex size-8 items-center justify-center rounded-full bg-secondary text-xs font-bold text-secondary-foreground">MC</div><div className="hidden xl:block"><p className="text-xs font-semibold leading-4">Maya Chen</p><p className="text-[10px] text-muted-foreground">Researcher</p></div><ChevronDown className="hidden size-3 text-muted-foreground xl:block" /></div>
            </div>
          </header>
          {helpOpen && <div className="border-b border-primary/20 bg-secondary/70 px-6 py-3 text-xs text-secondary-foreground"><div className="mx-auto flex max-w-[1440px] items-center justify-between gap-4"><span><strong>Research workspace guide:</strong> all visible recordings, metrics, and model activity are illustrative mock data. No live model connection is configured.</span><Button variant="ghost" size="sm" aria-label="Dismiss guide" onClick={() => setHelpOpen(false)}><X /></Button></div></div>}
          <main className="mx-auto w-full max-w-[1600px] flex-1 px-4 py-6 md:px-7 md:py-8">{children}</main>
          <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-5 py-3 text-[10px] text-muted-foreground md:px-7"><span>Duplex Lab <span className="px-1.5">·</span> Research prototype</span><span className="flex items-center gap-1.5"><span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />Live Research Pipeline Active</span></footer>
        </div>
      </div>
      <Toaster position="bottom-right" richColors />
    </TooltipProvider>
  );
}

export function MockNotice({ children = "Full-Duplex Pipeline Active · Live Engine" }: { children?: ReactNode }) {
  return <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-[10px] font-semibold text-emerald-700"><span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />{children}</span>;
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description: string; actions?: ReactNode }) {
  return <div className="mb-7 flex flex-col justify-between gap-4 lg:flex-row lg:items-end"><div className="min-w-0">{eyebrow && <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.14em] text-primary">{eyebrow}</p>}<h1 className="font-display text-2xl font-bold leading-tight md:text-[28px]">{title}</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-muted-foreground">{description}</p></div>{actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}</div>;
}

export function cardClass(extra = "") { return `rounded-xl border border-border bg-card shadow-[0_2px_12px_-6px_color-mix(in_oklab,var(--foreground)_12%,transparent)] ${extra}`; }
export function SectionTitle({ title, detail, action }: { title: string; detail?: string; action?: ReactNode }) {
  return <div className="mb-4 flex items-center justify-between gap-3"><div><h2 className="text-sm font-semibold">{title}</h2>{detail && <p className="mt-1 text-xs text-muted-foreground">{detail}</p>}</div>{action}</div>;
}

export function notify(message: string) { toast.success(message); }