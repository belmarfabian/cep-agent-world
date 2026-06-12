"""Script principal para ejecutar simulaciones del mundo CEP."""

import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
from rich.console import Console
from rich.table import Table

from src.data.loader import load_cep
from src.data.profiler import build_profiles
from src.data.clustering import find_archetypes, assign_archetypes
from src.data.distributions import extract_opinion_distributions
from src.agents.abm_agent import ABMAgent
from src.agents.llm_agent import LLMAgent
from src.agents.hybrid_agent import HybridAgent
from src.world.topology import build_network
from src.world.abm_world import ABMWorld
from src.world.llm_world import LLMWorld
from src.world.hybrid_world import HybridWorld
from src.analysis.metrics import polarization_variance, esteban_ray_index
from src.analysis.visualization import (
    plot_opinion_evolution,
    plot_opinion_distribution,
    plot_network,
)
from src.utils.config import SimulationConfig

console = Console()


def run_abm(config: SimulationConfig):
    """Opción B: simulación ABM clásica."""
    console.print("\n[bold green]=== OPCION B: ABM Clasico ===[/bold green]\n")

    # Cargar datos
    console.print("Cargando datos CEP...")
    df = load_cep(config.data_path(config.cep_data_path))
    profiles = build_profiles(df, max_agents=config.n_agents)
    console.print(f"  {len(profiles)} perfiles generados")

    # Crear agentes ABM
    agents = [
        ABMAgent(p, epsilon=config.epsilon, mu=config.mu, noise_sigma=config.noise_sigma)
        for p in profiles
    ]

    # Inicializar opiniones para los temas activos
    for agent in agents:
        for topic in ["pensiones", "seguridad", "inmigracion", "gobierno"]:
            if topic not in agent.profile.current_opinions:
                # Derivar opinión inicial del eje izq-der + ruido
                base = agent.profile.current_opinions.get("eje_izq_der", 0.0)
                noise = np.random.default_rng(hash(agent.agent_id) % 10000).normal(0, 0.2)
                agent.profile.current_opinions[topic] = np.clip(base + noise, -1, 1)

    # Construir red
    console.print(f"Construyendo red {config.network_type}...")
    network = build_network(agents, config.network_type, k=config.sw_k, p=config.sw_p, seed=config.random_seed)
    console.print(f"  {network.number_of_nodes()} nodos, {network.number_of_edges()} aristas")

    # Ejecutar
    world = ABMWorld(agents, network, config)
    console.print(f"Ejecutando {config.n_steps} pasos...")
    history = world.run_with_progress(config.n_steps)

    # Resultados
    _print_results(world, history, config)
    return world, history


def run_llm(config: SimulationConfig):
    """Opción A: simulación con agentes LLM."""
    console.print("\n[bold blue]=== OPCION A: Agentes LLM ===[/bold blue]\n")

    if not config.anthropic_api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY no configurada en .env[/red]")
        return None, None

    df = load_cep(config.data_path(config.cep_data_path))
    profiles = build_profiles(df, max_agents=config.n_agents)
    agents = [LLMAgent(p) for p in profiles]

    network = build_network(agents, config.network_type, k=config.sw_k, p=config.sw_p, seed=config.random_seed)

    world = LLMWorld(agents, network, config)
    console.print(f"Ejecutando {config.n_steps} pasos (con llamadas a Claude API)...")
    history = world.run(config.n_steps)

    console.print(f"Costo estimado: USD {world.client.estimated_cost_usd:.2f}")
    world.close()

    _print_results(world, history, config)
    return world, history


def run_hybrid(config: SimulationConfig):
    """Opción C: simulación híbrida LLM + estadística."""
    console.print("\n[bold magenta]=== OPCION C: Hibrido ===[/bold magenta]\n")

    if not config.anthropic_api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY no configurada en .env[/red]")
        return None, None

    df = load_cep(config.data_path(config.cep_data_path))
    profiles = build_profiles(df, max_agents=config.n_agents)

    # Clustering
    console.print(f"Generando {config.n_archetypes} arquetipos...")
    arch_result = find_archetypes(df, n_clusters=config.n_archetypes, random_state=config.random_seed)
    profiles = assign_archetypes(profiles, arch_result)

    console.print(f"  Silhouette score: {arch_result['silhouette']:.3f}")
    for k, v in arch_result["archetypes"].items():
        console.print(f"  Arquetipo {k}: {v['name']} ({v['n_respondents']} respondentes)")

    # Crear agentes híbridos
    agents = []
    for p in profiles:
        arch_id = p.archetype_id or 0
        arch = arch_result["archetypes"][arch_id]
        agents.append(HybridAgent(
            profile=p,
            archetype_centroid=arch["centroid"],
            archetype_std=arch["std"],
            drift_correction=config.drift_correction_strength,
        ))

    network = build_network(agents, config.network_type, k=config.sw_k, p=config.sw_p, seed=config.random_seed)
    world = HybridWorld(agents, network, config, arch_result)

    console.print(f"Ejecutando {config.n_steps} pasos...")
    history = world.run(config.n_steps)

    console.print(f"Costo estimado: USD {world.client.estimated_cost_usd:.2f}")
    world.close()

    _print_results(world, history, config)
    return world, history


def _print_results(world, history, config):
    """Imprime tabla de resultados finales."""
    console.print("\n[bold]Resultados finales:[/bold]")

    table = Table(title="Metricas de opinion")
    table.add_column("Tema", style="cyan")
    table.add_column("Media", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Polarizacion", justify="right")

    topics = ["eje_izq_der", "eval_gobierno", "democracia"]
    for topic in topics:
        opinions = world.get_all_opinions(topic)
        nonzero = [o for o in opinions if o != 0.0]
        if nonzero:
            table.add_row(
                topic,
                f"{np.mean(nonzero):.3f}",
                f"{np.std(nonzero):.3f}",
                f"{polarization_variance(nonzero):.3f}",
            )

    console.print(table)

    # Guardar resultados
    output_path = config.data_path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / f"history_{config.simulation_type}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)

    snapshot = world.get_snapshot()
    with open(output_path / f"snapshot_{config.simulation_type}.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)

    console.print(f"\n[green]Resultados guardados en {output_path}/[/green]")

    # Generar gráficos
    try:
        fig = plot_opinion_evolution(history, topics, title=f"Simulación {config.simulation_type.upper()}")
        fig.savefig(output_path / f"evolution_{config.simulation_type}.png", bbox_inches="tight")
        plt.close(fig)

        for topic in topics:
            fig = plot_opinion_distribution(snapshot, topic, title=f"{topic} — {config.simulation_type}")
            fig.savefig(output_path / f"dist_{topic}_{config.simulation_type}.png", bbox_inches="tight")
            plt.close(fig)

        fig = plot_network(world.network, world.agents)
        fig.savefig(output_path / f"network_{config.simulation_type}.png", bbox_inches="tight")
        plt.close(fig)

        console.print("[green]Gráficos guardados.[/green]")
    except Exception as e:
        console.print(f"[yellow]No se pudieron generar gráficos: {e}[/yellow]")


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Simulación de agentes CEP Chile")
    parser.add_argument("--type", choices=["abm", "llm", "hybrid", "all"], default="abm",
                        help="Tipo de simulación")
    parser.add_argument("--agents", type=int, default=100, help="Número de agentes")
    parser.add_argument("--steps", type=int, default=50, help="Número de pasos")
    parser.add_argument("--network", choices=["small_world", "homophily", "scale_free"],
                        default="small_world", help="Tipo de red")
    args = parser.parse_args()

    config = SimulationConfig(
        simulation_type=args.type if args.type != "all" else "abm",
        n_agents=args.agents,
        n_steps=args.steps,
        network_type=args.network,
    )

    console.print("[bold]=== Mundo Simulado de Agentes - CEP Chile ===[/bold]")

    if args.type == "all":
        run_abm(config)
        config.simulation_type = "llm"
        run_llm(config)
        config.simulation_type = "hybrid"
        run_hybrid(config)
    elif args.type == "abm":
        run_abm(config)
    elif args.type == "llm":
        run_llm(config)
    elif args.type == "hybrid":
        run_hybrid(config)
