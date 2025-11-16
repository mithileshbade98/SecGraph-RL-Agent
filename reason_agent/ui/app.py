"""
Streamlit UI for SecGraph-RL Agent.

5 tabs:
1. Query - Run queries and see reasoning traces
2. Graph - Inspect temporal graph snapshots
3. RL Training - Run PPO/DPO training
4. Audits - Create preference pairs
5. Drift - Monitor performance drift
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="SecGraph-RL Agent",
    page_icon="🔒",
    layout="wide",
)

st.title("🔒 SecGraph-RL Agent: Multi-Account Abuse Detection")
st.caption("Security-grade RL agent with graph reasoning and verifiable rewards")

# Initialize components (cached)
@st.cache_resource
def load_components():
    from reason_agent.tools.registry import ToolRegistry
    from reason_agent.tools.router import ToolRouter
    from reason_agent.tools.executors import ExecutorRegistry
    from reason_agent.reasoning.planner import ReasoningPlanner
    from reason_agent.reasoning.trace_recorder import TraceRecorder
    from reason_agent.rl.rewards import RewardComputer

    registry = ToolRegistry()
    router = ToolRouter(registry, method="rule")
    executors = ExecutorRegistry()
    planner = ReasoningPlanner(registry, router, executors, use_llm=False)
    trace_recorder = TraceRecorder()
    reward_computer = RewardComputer({
        'process_weight': 0.3,
        'final_weight': 0.7,
    })

    return {
        'registry': registry,
        'planner': planner,
        'trace_recorder': trace_recorder,
        'reward_computer': reward_computer,
    }

components = load_components()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Query",
    "🕸️ Graph",
    "🎓 RL Training",
    "✅ Audits",
    "📊 Drift"
])

# Tab 1: Query
with tab1:
    st.header("Query Agent")

    query = st.text_input(
        "Enter query:",
        value="Detect multi-account abuse with shared devices",
        key="query_input"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_query = st.button("🚀 Run Query", type="primary")

    if run_query and query:
        with st.spinner("Planning and executing..."):
            # Execute query
            trace = components['planner'].plan_and_execute(query)

            # Compute rewards
            rewards = components['reward_computer'].compute_reward(trace, [])

            # Display results
            st.success(f"Query completed: {'✅ Success' if trace['success'] else '❌ Failed'}")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Steps", trace['num_steps'])
            col2.metric("Process Reward", f"{rewards['process_reward']:.2f}")
            col3.metric("Final Reward", f"{rewards['final_reward']:.2f}")
            col4.metric("Total Reward", f"{rewards['total_reward']:.2f}")

            # Reasoning steps
            st.subheader("Reasoning Trace")
            for step in trace['steps']:
                with st.expander(f"Step {step['step_id']}: {step['thought']}"):
                    st.write(f"**Tool:** {step['tool']}")
                    st.write(f"**Parameters:** {step['parameters']}")
                    st.json(step.get('result', {}))

            # Full trace JSON
            st.subheader("Full Trace")
            st.json({**trace, 'rewards': rewards})

# Tab 2: Graph
with tab2:
    st.header("Graph Inspector")

    col1, col2 = st.columns(2)
    with col1:
        as_of_date = st.date_input("As-of date")
    with col2:
        limit = st.number_input("Result limit", value=100, min_value=1, max_value=1000)

    if st.button("Query Graph Snapshot"):
        st.info("Graph snapshot query (requires Neo4j connection)")
        # Mock data
        st.write("**Node counts:**")
        st.json({
            'User': 150,
            'Device': 45,
            'Email': 160,
            'IP': 80,
            'Session': 500,
        })

        st.write("**Shared device clusters:**")
        st.dataframe(pd.DataFrame([
            {'device_id': 'device_123', 'accounts': 5, 'anomaly_score': 0.92},
            {'device_id': 'device_456', 'accounts': 3, 'anomaly_score': 0.78},
        ]))

# Tab 3: RL Training
with tab3:
    st.header("RL Training")

    train_col1, train_col2 = st.columns(2)

    with train_col1:
        st.subheader("PPO Training")
        st.write("Train with verifiable rewards on math tasks")
        st.info("Model: TinyLlama-1.1B-Chat-v1.0 (downloads automatically if not cached)")
        ppo_episodes = st.number_input("Episodes", value=10, min_value=1, max_value=100, key="ppo_ep")

        if st.button("Train PPO"):
            from reason_agent.rl.ppo import PPOTrainer
            try:
                with st.spinner("Initializing model and trainer..."):
                    trainer = PPOTrainer(
                        initialize_model=True,
                        use_value_head=True,
                    )

                with st.spinner(f"Training PPO for {ppo_episodes} episodes..."):
                    metrics = trainer.train(num_episodes=ppo_episodes)

                st.success(f"Training complete! Final reward: {metrics['final_avg_reward']:.3f}")
                st.line_chart(pd.DataFrame({
                    'Episode': range(len(metrics['mean_rewards'])),
                    'Reward': metrics['mean_rewards']
                }).set_index('Episode'))
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.info("Make sure the model is downloaded. Run: python3 scripts/download_model.py")

    with train_col2:
        st.subheader("DPO Training")
        st.write("Train with preference pairs from audits")
        st.info("Model: TinyLlama-1.1B-Chat-v1.0 (downloads automatically if not cached)")
        dpo_epochs = st.number_input("Epochs", value=1, min_value=1, max_value=10, key="dpo_ep")

        if st.button("Train DPO"):
            from reason_agent.rl.dpo import DPOTrainer
            try:
                with st.spinner("Initializing model and trainer..."):
                    trainer = DPOTrainer(
                        initialize_model=True,
                    )

                with st.spinner(f"Training DPO for {dpo_epochs} epochs..."):
                    metrics = trainer.train(num_epochs=dpo_epochs)

                st.success(f"Training complete! Accuracy: {metrics['final_accuracy']:.3f}")
                st.write(f"Trained on {metrics['num_pairs']} preference pairs")
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.info("Make sure the model is downloaded and preference pairs exist in data/audits/pairs.jsonl")

# Tab 4: Audits
with tab4:
    st.header("Expert Audits")
    st.write("Review trace pairs and select preferred reasoning")

    # Load recent traces
    traces = components['trace_recorder'].load_recent_traces(n=10)

    if len(traces) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Trace A")
            trace_a = traces[0]
            st.write(f"**Query:** {trace_a['query']}")
            st.write(f"**Steps:** {len(trace_a.get('steps', []))}")
            st.json(trace_a.get('steps', []))

        with col2:
            st.subheader("Trace B")
            trace_b = traces[1]
            st.write(f"**Query:** {trace_b['query']}")
            st.write(f"**Steps:** {len(trace_b.get('steps', []))}")
            st.json(trace_b.get('steps', []))

        st.subheader("Preference")
        preference = st.radio("Which trace is better?", ["A", "B", "Equal"])
        rationale = st.text_area("Rationale:")

        if st.button("Submit Audit"):
            # Save preference pair
            pair = {
                'prompt': trace_a['query'],
                'chosen': trace_a if preference == "A" else trace_b,
                'rejected': trace_b if preference == "A" else trace_a,
                'rationale': rationale,
            }

            output_file = Path("data/audits/pairs.jsonl")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'a') as f:
                f.write(json.dumps(pair) + '\n')

            st.success("Audit saved to data/audits/pairs.jsonl")
    else:
        st.info("Need at least 2 traces for comparison. Run queries in Query tab.")

# Tab 5: Drift
with tab5:
    st.header("Drift Monitoring")

    if st.button("Analyze Drift"):
        from reason_agent.monitoring.drift import DriftDetector

        detector = DriftDetector()
        traces = components['trace_recorder'].load_recent_traces(n=100)

        if traces:
            # Tool success rates
            tool_stats = detector.analyze_tool_success_rate(traces)
            st.subheader("Tool Success Rates")
            st.bar_chart(pd.DataFrame.from_dict(
                tool_stats['tool_success_rates'],
                orient='index',
                columns=['Success Rate']
            ))

            if tool_stats['low_success_tools']:
                st.warning(f"⚠️ Low success tools: {tool_stats['low_success_tools']}")

            # Reward trend
            rewards = [t.get('rewards', {}).get('total_reward', 0.0) for t in traces]
            reward_stats = detector.compute_reward_ewma(rewards)

            st.subheader("Reward EWMA Trend")
            st.line_chart(pd.DataFrame({
                'Trace': range(len(reward_stats['ewma_history'])),
                'EWMA': reward_stats['ewma_history']
            }).set_index('Trace'))

            if reward_stats['drop_detected']:
                st.error(f"⚠️ Reward drop detected! Magnitude: {reward_stats['drop_magnitude']:.3f}")

        else:
            st.info("No traces available for analysis")
