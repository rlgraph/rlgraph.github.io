---
layout: default
title:  "RLgraph: A unified interface for design and execution of RL algorithms"
date:   2019-01-04 14:41:37 +0100
categories: rlgraph
---
# RLgraph: A unified interface for design and execution of RL algorithms
We introduce RLgraph, a RL framework decoupling logical component composition from deep learning backend and distributed execution. 

Using RLgraph, users combine high level components in a space-independent manner and define input spaces. RLgraph then builds a backend-independent component graph which can be translated to a TensorFlow computation graph, or executed in define by run mode via PyTorch (1.0+).

 The resulting agents can be trained locally embedded in an application like any other library (e.g. numpy), or using backends such as distributed TensorFlow, Ray, and Horovod. 

![RLgraph stack]({{ site.url }}/images/rlgraph_stack.png)

This design has numerous advantages compared to many existing libraries:

- Each component can be built and tested as its own graph - being able to systematically test arbitrary sub-graphs of complex algorithms drastically accelerates debugging and prototyping. Think of components as Sonnet-style objects but including API and device management, build system, separation of variable creation and computation logic.
- Backend and space-independent high-level logic enables fast exploration of new designs.
- Future-proof: Developers can decide on a per-model basis which backend to use to access specific framework features. Components can plug into any external library (e.g. TRFL).
- Scalability: RLgraph agents can be arbitrarily executed e.g. on Ray using our Ray execution package, using distributed TensorFlow to explore end-to-end graphs, or using any distribution mechanism of choice. 
- No platform lock-in: The component graph is discarded after building for a specific backend, and models can be exported to framework specific standard formats.

## Architecture

Users interact with RLgraph agents through a front-end agent API very similar to TensorForce (which some of us also created/worked on). Internally, agents rely on a graph executor which serves requests against the component graph and manages backend-specific execution semantics. For example,the TensorFlowExecutor handles sessions, devices, summaries, placeholders, profiling, distributed TF servers, and timelines. The PyTorch executor in turn manages default tensor types, devices, or external plugins such as Horovod.

![RLgraph execution]({{ site.url }}/images/rlgraph_graph_execution.png)


This agent can be executed on external engines (e.g. using RLgraph's ```RayExecutors```) or be used like any other object in an application. 

## API examples 

Most applied users will be able to rely on a comprehensive high-level API. We show how agents can be configured and executed seamlessly in local, parallel, and distributed contexts:

```python
from rlgraph.agents import agents
from rlgraph.spaces import *

# Describe inputs via powerful space objects which manage batch and time ranks.
states = IntBox(
	low=0,
	high=1024,
	add_time_rank=True
)

# Nested container spaces are used throughout RLgraph
# to describe state and perform shape inference.
actions = Dict(
	int_action=IntBox(low=0, high=5),
	bool_action=BoolBox(),
	float_action=FloatBox(shape=(3,)
)

# Create PPO agent
agent = Agent.from_spec(
	# TODO how much to show from config? Bloats screen.
        agent_config,
        state_space=env.state_space,
        action_space=env.action_space
)

# Use agent locally, control behaviour with flags.
actions = agent.get_actions(states, use_exploration=use_exploration, apply_preprocessing=True)

# Batch observe multi-environment settings.
agent.observe(states, actions, rewards, terminals, batched=True , env_id="env_3")

# Updates by sampling from buffer after observing.
loss = agent.update()

# Update from an external batch which may contain arbitrary (non-terminal) sub-episode fragments from multiple environments, identified via sequence indices:
agent.update(batch=dict(states=states, actions=actions, rewards=rewards, terminals=terminals, sequence_indices=sequence_indices)

# Single-threaded, vectorized execution. 
env_spec = {"type": "openai", "env_id": "CartPole-v0"}
worker = SingleThreadedWorker(env_spec=env_spec, agent=agent, num_envs=8)
stats = worker.execute_timesteps(100000)

# Go large scale with Ray in 2 lines.
ray_executor = SyncBatchExecutor(agent_config, env_spec)
ray_stats = ray_executor.execute(steps=100000)

# Use agent trained on Ray just as before.
agent = ray_executor.local_agent
```
Full [example scripts](https://github.com/rlgraph/rlgraph/tree/master/examples) and configuration files can be found in the repository. 

## Components

Everything in RLgraph is a ```Component```. An agent is a component containing sub-components like memories, neural networks, loss functions, or optimizers. Components interact with each other through API functions which create data-flow between components. That is, using the TensorFlow backend, RLgraph creates end-to-end static graphs by stichting together API calls by tracing them during the build procedure. For example, below is a simple API method to request actions:

```python
# TODO not a good example because not clear why API necessary.
@rlgraph_api(component=self.root_component)
def action_from_preprocessed_state(self, preprocessed_states,
 										  time_step=0, use_exploration=True):
 	# Policy and exploration components expose API methods.
    sample_deterministic = policy.get_deterministic_action(preprocessed_states)
    actions = exploration.get_action(sample_deterministic["action"], time_step, use_exploration)
    return preprocessed_states, actions
```

API method decorators wrap these functions to create end-to-end dataflow. RLgraph manages session, variablestates, devices, scopes, placeholders, nesting, time and batchranks for each component and its incoming and leaving dataflow.

This means we can build a component from a space and interact with it without needing to manually create tensors, input placeholders etc:

```python
record_space = Dict(
    states=dict(state1=float, state2=float),
    actions=dict(action1=float, action2=IntBox(10)),
    # For scalar spaces, use Python primitives
    reward=float,
    terminals=BoolBox(),
    next_states=dict(state1=float, state2=float),
    add_batch_rank=True
)

# Memory exposes insert, sample..methods
memory = ReplayMemory(capacity=1000) 

# Input spaces contain spaces for all arguments.
input_spaces = dict(records=record_space, num_records=int)

# Builds the memory with variables, placeholders for these spaces.
graph = ComponentTest(component=memory, input_spaces=input_spaces)

# Generate a sample batch from the nested space.
observation = record_space.sample(size=32)

# Calls session, fetches ops, prepares inputs, executes API method.
graph.test(("insert_records", observation))

# Get some samples providing the "num_records" int-arg.
samples = graph.test(("get_records", 10))
```

Separating spaces of tensors from logical composition enables us to reuse components without ever manually dealing with incompatible shapes again. Note how the above code does not contain any framework-specific notions but only defines an input dataflow from a set of spaces. 

When composing components, the build process performs shape inference to detect how a component's API methods transform its input spaces so we can compute the correct inputs to the next component to ensure down-stream variables and placeholders have the correct shapes.

In future blog-posts, we will show in more detail how to implement custom components and compotatuons, and how to plug in external libraries to prototype new algorithms with RLgraph.

## Resources and road-map

RLgraph is in alpha stage and is currently being piloted in various research projects. We welcome contributions and feedback. Over the next few months, we will keep building out utilities (additional Ray executors, better device management for PyTorch, ONNX integration and more), and implement more algorithms. Feel free to create issues discussing improvements.

Code: Get started building applications with RLgraph: [https://github.com/rlgraph/rlgraph](https://github.com/rlgraph/rlgraph)

Documentation: Docs are available on [readthedocs](https://rlgraph.readthedocs.io/en/latest/?badge=latest)

Paper: Read our [paper](https://arxiv.org/abs/1810.09028) to learn more about RLgraph's design. 

