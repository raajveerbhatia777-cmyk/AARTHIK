import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TextArea, Button, Static, Label
from textual.containers import Vertical, Horizontal
import httpx

class PolicyAnalyzerApp(App):
    CSS = """
    Screen {
        layout: vertical;
        padding: 1 2;
    }
    #input-box {
        height: 8;
        margin-bottom: 1;
    }
    #analyze-btn {
        width: 100%;
        margin-bottom: 1;
    }
    #results-card {
        border: solid green;
        padding: 1;
        height: 12;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("[b]Enter Policy Statement for Analysis:[/b]")
        yield TextArea("This is a live test of our terminal UI engine.", id="input-box")
        yield Button("Analyze Stance", id="analyze-btn", variant="primary")
        yield Label("[b]Analysis Results:[/b]")
        yield Static("Waiting for input...", id="results-card")
        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "analyze-btn":
            text_area = self.query_one("#input-box", TextArea)
            results_box = self.query_one("#results-card", Static)
            statement = text_area.text.strip()

            if not statement:
                results_box.update("[red]Error: Please enter a policy statement.[/red]")
                return

            results_box.update("[yellow]Analyzing...[/yellow]")
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://127.0.0.1:8000/api/v1/analyze-stance/json",
                        json={"statement_text": statement},
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        formatted_res = (
                            f"[bold green]Score:[/bold green] {data.get('score')}\n"
                            f"[bold cyan]Tone:[/bold cyan] {data.get('tone')}\n"
                            f"[bold yellow]Summary:[/bold yellow] {data.get('summary')}\n\n"
                            f"[bold magenta]Key Findings:[/bold magenta] {', '.join(data.get('key_findings', []))}\n"
                            f"[dim]Stub Mode: {data.get('details', {}).get('stub')}[/dim]"
                        )
                        results_box.update(formatted_res)
                    else:
                        results_box.update(f"[red]API Error: Status {response.status_code}[/red]")
            except Exception as e:
                results_box.update(f"[red]Failed to connect to API: {e}[/red]")

if __name__ == "__main__":
    app = PolicyAnalyzerApp()
    app.run()
