"""
fg_sync/cli.py
--------------
Click CLI entry point for fg-sync.

Commands:
  fg-sync run          Start proxy + scheduler daemon (blocks)
  fg-sync sync         One-shot pipeline run then exit
  fg-sync status       Show current ruleset and injector state
  fg-sync metrics      Show M1-M5 comparison table
  fg-sync export       Export prompt prefix as plain text
  fg-sync reset        Clear capture log, cursor, and ruleset
  fg-sync init         Generate fg-sync.toml in ~/.fg-sync/

Run `fg-sync --help` or `fg-sync <command> --help` for details.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import click
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from fg_sync.config import load_config, FG_SYNC_HOME, DEFAULT_CONFIG_PATH
from fg_sync.injector import Injector

console = Console()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING if not verbose else logging.DEBUG)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--config", "-c", default=None, help="Path to fg-sync.toml")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, config: str | None, verbose: bool):
    """
    fg-sync — Fractal Grammar ↔ Ollama integration CLI sidecar.

    Persistent behavioral memory for local LLMs. No cloud. No fine-tuning.

    \b
    Quick start:
      fg-sync init         # create config
      fg-sync run          # start proxy + daemon
      fg-sync status       # check state after first sync
    """
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------------------
# fg-sync run
# ---------------------------------------------------------------------------

@main.command()
@click.option("--source", type=click.Choice(["proxy", "openwebui"]), default=None,
              help="Override source type from config")
@click.option("--no-proxy", is_flag=True, default=False,
              help="Run scheduler only — do not start HTTP proxy")
@click.pass_context
def run(ctx: click.Context, source: str | None, no_proxy: bool):
    """
    Start the fg-proxy HTTP interceptor and the pipeline scheduler daemon.

    The proxy listens on localhost:11435 (configurable) and forwards all
    requests to Ollama at localhost:11434. Conversation data is captured
    to ~/.fg-sync/capture.jsonl.

    The scheduler runs the fractal-grammar pipeline on the configured cron
    schedule (default: 2am UTC nightly) and hot-reloads the behavioral
    ruleset on each run.

    \b
    Point your Ollama client at port 11435 instead of 11434.
    Everything else works identically.
    """
    cfg = load_config(ctx.obj.get("config_path"))
    cfg.ensure_dirs()

    if source:
        cfg.source.type = source

    injector = Injector(
        ruleset_path=cfg.ruleset.output_path,
        max_prompt_tokens=cfg.ruleset.max_prompt_tokens,
    )

    console.print(Panel.fit(
        f"[bold green]fg-sync v0.1.0[/bold green]\n"
        f"Source: [cyan]{cfg.source.type}[/cyan]\n"
        f"Proxy: [cyan]localhost:{cfg.proxy.listen_port}[/cyan] → Ollama [cyan]localhost:{cfg.proxy.ollama_port}[/cyan]\n"
        f"Schedule: [cyan]{cfg.pipeline.schedule}[/cyan] (UTC)\n"
        f"Ruleset: [cyan]{cfg.ruleset.output_path}[/cyan]\n"
        f"Injector: [{'green]active' if injector.is_active() else 'yellow]no ruleset yet — run sync first'}[/]",
        title="fg-sync",
        border_style="green",
    ))

    if no_proxy:
        # Run scheduler only
        _run_daemon_only(cfg, injector)
    else:
        # Run proxy + daemon concurrently
        _run_proxy_and_daemon(cfg, injector)


def _run_proxy_and_daemon(cfg, injector):
    """Run proxy (uvicorn) in a thread + daemon scheduler in main thread."""
    import threading
    from fg_sync.proxy import FgProxy
    from fg_sync.daemon import FgSyncDaemon

    proxy = FgProxy(config=cfg.proxy, injector=injector)
    daemon = FgSyncDaemon(config=cfg, injector=injector)

    # Start uvicorn in a background thread
    uv_config = uvicorn.Config(
        app=proxy.app,
        host="127.0.0.1",
        port=cfg.proxy.listen_port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(uv_config)

    proxy_thread = threading.Thread(target=server.run, daemon=True, name="fg-proxy")
    proxy_thread.start()

    console.print(f"[green]✓[/green] fg-proxy started on port {cfg.proxy.listen_port}")
    console.print(f"[dim]  Point your Ollama client at: http://localhost:{cfg.proxy.listen_port}[/dim]")
    console.print(f"[green]✓[/green] Pipeline scheduler starting...")

    try:
        daemon.start()
    finally:
        server.should_exit = True


def _run_daemon_only(cfg, injector):
    from fg_sync.daemon import FgSyncDaemon
    daemon = FgSyncDaemon(config=cfg, injector=injector)
    console.print("[green]✓[/green] Running pipeline scheduler only (no proxy)")
    daemon.start()


# ---------------------------------------------------------------------------
# fg-sync sync
# ---------------------------------------------------------------------------

@main.command()
@click.option("--dry-run", is_flag=True, default=False,
              help="Run pipeline without writing ruleset or advancing cursor")
@click.option("--source", type=click.Choice(["proxy", "openwebui"]), default=None)
@click.pass_context
def sync(ctx: click.Context, dry_run: bool, source: str | None):
    """
    Run the fractal-grammar pipeline once and exit.

    Reads new captures from capture.jsonl since the last cursor position,
    runs the full pipeline, and writes a new ruleset.json.

    Use this for manual syncs or when running fg-sync from a system crontab
    instead of the built-in scheduler.

    \b
    Example crontab entry:
      0 2 * * * /usr/local/bin/fg-sync sync >> ~/.fg-sync/logs/cron.log 2>&1
    """
    from fg_sync.daemon import run_once

    config_path = ctx.obj.get("config_path")
    cfg = load_config(config_path)
    if source:
        cfg.source.type = source

    if dry_run:
        console.print("[yellow]DRY RUN — no files will be written[/yellow]")

    with console.status("[bold]Running fractal-grammar pipeline...[/bold]"):
        produced = run_once(config_path=config_path, dry_run=dry_run)

    if produced:
        console.print("[green]✓[/green] Pipeline complete — ruleset.json updated")
        console.print(f"  [dim]Run [bold]fg-sync status[/bold] to inspect the new ruleset[/dim]")
    else:
        console.print("[yellow]⚠[/yellow] Pipeline skipped — not enough new captures or no new data")
        capture_path = cfg.proxy.capture_path
        if capture_path.exists():
            import os
            size = os.path.getsize(capture_path)
            console.print(f"  [dim]capture.jsonl: {size:,} bytes[/dim]")
            console.print(f"  [dim]Need {cfg.pipeline.min_events_to_run} user messages to trigger pipeline[/dim]")
        else:
            console.print(f"  [dim]No capture.jsonl found. Start the proxy and have some conversations first.[/dim]")


# ---------------------------------------------------------------------------
# fg-sync status
# ---------------------------------------------------------------------------

@main.command()
@click.pass_context
def status(ctx: click.Context):
    """
    Show current ruleset state and injector configuration.
    """
    cfg = load_config(ctx.obj.get("config_path"))

    injector = Injector(
        ruleset_path=cfg.ruleset.output_path,
        max_prompt_tokens=cfg.ruleset.max_prompt_tokens,
    )
    s = injector.status()

    if not s["active"]:
        console.print(Panel(
            f"[yellow]No ruleset loaded.[/yellow]\n\n"
            f"Run [bold]fg-sync sync[/bold] to generate your first behavioral ruleset.\n"
            f"You need at least [bold]{cfg.pipeline.min_events_to_run}[/bold] conversation messages captured first.\n\n"
            f"Ruleset path: [dim]{s['ruleset_path']}[/dim]",
            title="fg-sync status",
            border_style="yellow",
        ))
        return

    # Summary panel
    console.print(Panel(
        f"[green]Active[/green] — behavioral memory injected into every Ollama request\n\n"
        f"Generated:   [cyan]{s['generated_at']}[/cyan]\n"
        f"Events:      [cyan]{s['event_count']:,}[/cyan]\n"
        f"Sessions:    [cyan]{s['session_count']:,}[/cyan]\n"
        f"Rules:       [cyan]{s['rule_count']}[/cyan]\n"
        f"Prefix:      [cyan]{s['prompt_prefix_chars']} chars (~{s['prompt_prefix_chars']//4} tokens)[/cyan]",
        title="fg-sync status",
        border_style="green",
    ))

    # Rules table
    if s.get("top_rules"):
        table = Table(title="Top Behavioral Rules", show_header=True, header_style="bold cyan")
        table.add_column("Label", style="white")
        table.add_column("Weight", justify="right", style="green")
        for rule in s["top_rules"]:
            table.add_row(rule["label"], f"{rule['weight']:.3f}")
        console.print(table)

    # Config summary
    console.print(f"\n[dim]Config:[/dim] schedule=[cyan]{cfg.pipeline.schedule}[/cyan]  "
                  f"source=[cyan]{cfg.source.type}[/cyan]  "
                  f"proxy=[cyan]:{cfg.proxy.listen_port}[/cyan]→[cyan]:{cfg.proxy.ollama_port}[/cyan]")


# ---------------------------------------------------------------------------
# fg-sync metrics
# ---------------------------------------------------------------------------

@main.group()
def metrics():
    """Performance metrics commands (M1–M5)."""
    pass


@metrics.command(name="compare")
@click.pass_context
def metrics_compare(ctx: click.Context):
    """Show M1–M5 comparison table: baseline vs fg-sync sessions."""
    from fg_sync.metrics import MetricsCollector

    cfg = load_config(ctx.obj.get("config_path"))
    collector = MetricsCollector(
        metrics_path=cfg.metrics.metrics_path,
        capture_path=cfg.proxy.capture_path,
        ruleset_path=cfg.ruleset.output_path,
    )
    console.print(collector.render_table())


@metrics.command(name="storage")
@click.pass_context
def metrics_storage(ctx: click.Context):
    """Show M3 storage footprint: capture.jsonl vs ruleset.json compression ratio."""
    from fg_sync.metrics import MetricsCollector

    cfg = load_config(ctx.obj.get("config_path"))
    collector = MetricsCollector(
        metrics_path=cfg.metrics.metrics_path,
        capture_path=cfg.proxy.capture_path,
        ruleset_path=cfg.ruleset.output_path,
    )
    report = collector.storage_report()

    table = Table(title="M3 — Storage Footprint", header_style="bold cyan")
    table.add_column("File", style="white")
    table.add_column("Size", justify="right", style="green")

    table.add_row("capture.jsonl (raw)", f"{report['capture_mb']:.3f} MB")
    table.add_row("ruleset.json", f"{report['ruleset_json_bytes']:,} B")
    table.add_row("assoc_memory.pkl", f"{report['assoc_memory_bytes']:,} B")
    table.add_row("[bold]Total fg-sync[/bold]", f"[bold]{report['fg_sync_kb']:.1f} KB[/bold]")
    table.add_row("[bold green]Compression Ratio[/bold green]",
                  f"[bold green]{report['compression_ratio']}[/bold green]")

    console.print(table)


# ---------------------------------------------------------------------------
# fg-sync export
# ---------------------------------------------------------------------------

@main.command()
@click.option("--format", "fmt", type=click.Choice(["txt", "json"]), default="txt",
              help="Output format: txt (plain system prompt) or json (full ruleset)")
@click.pass_context
def export(ctx: click.Context, fmt: str):
    """
    Export the current behavioral ruleset.

    txt: outputs the raw system prompt prefix (for use with any LLM tool)
    json: outputs the full ruleset.json
    """
    cfg = load_config(ctx.obj.get("config_path"))
    injector = Injector(
        ruleset_path=cfg.ruleset.output_path,
        max_prompt_tokens=cfg.ruleset.max_prompt_tokens,
    )

    if not injector.is_active():
        console.print("[yellow]No ruleset available. Run `fg-sync sync` first.[/yellow]")
        sys.exit(1)

    ruleset = injector.get_ruleset()

    if fmt == "txt":
        click.echo(ruleset.get("prompt_prefix", ""))
    elif fmt == "json":
        click.echo(json.dumps(ruleset, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# fg-sync reset
# ---------------------------------------------------------------------------

@main.command()
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def reset(ctx: click.Context, yes: bool):
    """
    Clear all fg-sync state: capture.jsonl, cursor.json, ruleset.json, assoc_memory.pkl.

    This does NOT affect Ollama or your conversation history — only fg-sync's
    learned behavioral data is removed.
    """
    cfg = load_config(ctx.obj.get("config_path"))
    from fg_sync.pipeline import CURSOR_FILE, ASSOC_MEMORY_FILE

    files_to_remove = [
        cfg.proxy.capture_path,
        cfg.ruleset.output_path,
        CURSOR_FILE,
        ASSOC_MEMORY_FILE,
        cfg.metrics.metrics_path,
    ]

    existing = [f for f in files_to_remove if f.exists()]

    if not existing:
        console.print("[green]Nothing to reset — fg-sync state is already clean.[/green]")
        return

    console.print("The following files will be deleted:")
    for f in existing:
        console.print(f"  [red]{f}[/red]")

    if not yes:
        click.confirm("\nThis cannot be undone. Continue?", abort=True)

    for f in existing:
        f.unlink()
        console.print(f"[dim]Deleted {f}[/dim]")

    console.print("[green]✓[/green] fg-sync state reset.")


# ---------------------------------------------------------------------------
# fg-sync init
# ---------------------------------------------------------------------------

@main.command()
@click.pass_context
def init(ctx: click.Context):
    """
    Generate a default fg-sync.toml in ~/.fg-sync/ and create required directories.
    """
    FG_SYNC_HOME.mkdir(parents=True, exist_ok=True)
    (FG_SYNC_HOME / "logs").mkdir(exist_ok=True)

    example_path = Path(__file__).parent.parent / "fg-sync.toml.example"
    target_path = DEFAULT_CONFIG_PATH

    if target_path.exists():
        console.print(f"[yellow]Config already exists at {target_path}[/yellow]")
        console.print("[dim]Edit it directly or run `fg-sync reset` to start fresh.[/dim]")
        return

    if example_path.exists():
        import shutil
        shutil.copy(example_path, target_path)
    else:
        # Write minimal default config inline
        target_path.write_text(_DEFAULT_CONFIG_CONTENT())

    console.print(f"[green]✓[/green] Config written to [bold]{target_path}[/bold]")
    console.print(f"\nNext steps:")
    console.print(f"  1. Edit [bold]{target_path}[/bold] if needed")
    console.print(f"  2. Run [bold]fg-sync run[/bold] to start the proxy and scheduler")
    console.print(f"  3. Point your Ollama client at [bold]http://localhost:11435[/bold]")
    console.print(f"  4. After {50} conversations, run [bold]fg-sync sync[/bold] manually or wait for nightly run")


def _DEFAULT_CONFIG_CONTENT() -> str:
    return """\
# fg-sync.toml — configuration for fg-sync
# See: https://github.com/ryandmoore1976/fractal-grammar

[proxy]
listen_port = 11435
ollama_port = 11434
ollama_host = "127.0.0.1"
capture_path = "~/.fg-sync/capture.jsonl"

[pipeline]
schedule = "0 2 * * *"          # nightly 2am UTC
min_events_to_run = 50
novelty_threshold = 0.92
hdc_dimensions = 10000
hdc_seed = 4277009102            # 0xFEEDBEEF
min_cluster_size = 5
assoc_memory_threshold = 0.05
use_hdc = true

[ruleset]
output_path = "~/.fg-sync/ruleset.json"
max_rules = 20
max_prompt_tokens = 400
recency_weight = 0.7
decay_days = 30

[metrics]
enabled = true
metrics_path = "~/.fg-sync/metrics.jsonl"
baseline_session_count = 10

[source]
type = "proxy"   # "proxy" (default) or "openwebui"
# openwebui_db = "~/.local/share/open-webui/webui.db"
"""


if __name__ == "__main__":
    main()
