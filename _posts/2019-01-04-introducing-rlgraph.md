---
layout: default
title:  "RLgraph: A unified interface for design and execution of RL algorithms"
date:   2019-01-04 14:41:37 +0100
categories: rlgraph
---
# RLgraph: Robust, incrementally testable reinforcement learning

We introduce RLgraph, a RL framework decoupling logical component composition from deep learning backend and distributed execution. RLgraph brings rigorous management of internal and external state, inputs, devices, and dataflow to reinforcement learning.

![APEX learning on multiple GPUs]({{ site.url }}/images/learning_combined.png)
<span class="caption">Left: Soft Actor Critic on Pendulum-v0 (10 seeds). Right: Multi-GPU Ape-X on Pong-v0 (10 seeds).</span>

Using RLgraph, developers combine high level components in a space-independent manner and define input spaces. RLgraph then builds a backend-independent component graph which can be translated to a TensorFlow computation graph, or executed in define by run mode via PyTorch (1.0+).

The resulting agents can be trained locally embedded in an application like any other library (e.g. numpy), or using backends such as distributed TensorFlow, Ray, and Horovod. RLgraph currently implements variants of contemporary methods such as SAC (thanks to a contributor), IMPALA, APE-X, PPO.

![RLgraph stack]({{ site.url }}/images/rlgraph_stack.png)

This design has a number of advantages compared to many existing implementations which struggle to resolve the tension between prototyping, reusability across backends, and scalable execution:

- Each component can be built and tested as its own graph - being able to systematically test arbitrary sub-graphs of complex algorithms drastically accelerates debugging and prototyping. Think of components as Sonnet-style objects but including API and device management, build system, separation of variable creation and computation logic. 
- Backend and space-independent high-level logic enables faster exploration of new designs.
- Scalability: RLgraph agents can be arbitrarily executed e.g. on Ray using our Ray execution package, using distributed TensorFlow to explore end-to-end graphs, or using any distribution mechanism of choice. 
- Maintenance: Extending existing frameworks often means either duplicating large amounts of code or intransparently piling on new learning heuristics into a single implementation. Neither is desirable. In RLgraph, heuristics are first-class citizens which are separately built and tested.

Overall, RLgraph targets an opinionated trade-off between enforcing strict interfaces and a rigorous build system, and exploration of new designs. Specifically, RLgraph is a framework meant for users that need to execute their research in practice in all kinds of deployment contexts. There is also a simple plug and play high level API (see below).

## Architecture

Users interact with RLgraph agents through a front-end agent API very similar to TensorForce (which some of us also created/worked on). Internally, agents rely on a graph executor which serves requests against the component graph and manages backend-specific execution semantics. For example,the TensorFlowExecutor handles sessions, devices, summaries, placeholders, profiling, distributed TF servers, and timelines. The PyTorch executor in turn manages default tensor types, devices, or external plugins such as Horovod.

![RLgraph execution]({{ site.url }}/images/rlgraph_graph_execution.png)

This agent can be executed on external engines (e.g. using RLgraph's ```RayExecutors```) or be used like any other object in an application. 

## API examples 

Most applied users will be able to rely on a comprehensive high-level API. We show how agents can be configured and executed seamlessly in local and distributed contexts:

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

# Update from an external batch which may contain 
# arbitrary (non-terminal) sub-episode fragments from multiple environments,
# identified via sequence indices:
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

Everything in RLgraph is a ```Component```. An agent implements an algorithm via a root component containing sub-components like memories, neural networks, loss functions, or optimizers. Components interact with each other through API functions which impliitly create data-flow between components. That is, using the TensorFlow backend, RLgraph creates end-to-end static graphs by stitching together API calls and tracing them during the build procedure. For example, below is a simple API method to update the policy network:

![RLgraph API example]({{ site.url }}/images/rlgraph_api_example.png)

API method decorators wrap API functions to create end-to-end dataflow. RLgraph manages session, variable/internal states, devices, scopes, placeholders, nesting, time and batchranks for each component and its incoming and outgoing dataflow.

We can build a component from a space and interact with it without needing to manually create tensors, input placeholders etc:

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
graph.test(memory.insert_records, observation))

# Get some samples providing the "num_records" int-arg.
samples = graph.test((memory.insert_records, 10))
```

Separating spaces of tensors from logical composition enables us to reuse components without ever manually dealing with incompatible shapes again. Note how the above code does not contain any framework-specific notions but only defines an input dataflow from a set of spaces. In RLgraph, heuristics (which often have great impact on performance in RL) are not afterthoughts but first class citizens which are tested both individually and in integration with other components. For example, the ```Policy``` component contains neural network, action adapter (and in turn layer and distribution) sub-components, all of which are tested separately.

The core difference between using RLgraph and standard implementation workflows is that every component is fully specified explicitly: Its devices and scopes for computations and internal states (e.g. variable sharing) are explicitly assigned which spares developers the headaches of nested context managers. As a result, RLgraph creates beautiful TensorBoard visualisations, e.g. our IMPALA implementation:

![RLgraph IMPALA tensorboard]({{ site.url }}/images/impala_tboard_graph_learner.png)

Compare with DeepMind's open source IMPALA using Sonnet and nested context managers:

<span class="image-scroll-container horizontal">
![Deepmind scalable agent IMPALA tensorboard]({{ site.url }}/images/impala_tboard_graph_learner_deepmind.png){:.scroll}
</span>


## Resources and call for contributions

RLgraph is in alpha stage and being used in a number of research pilots. We welcome contributions and feedback. As RLgraph ambitiously covers multiple frameworks and backends, there is a lot to do on all ends (we would especially welcome more PyTorch expertise).

Over the next few months, we will keep building out utilities (TF 2.0 considerations, separation of backend code (specifically see pinned issues), additional Ray executors, ...), and implement more algorithms. Feel free to create issues discussing improvements.

Code: Get started building applications with RLgraph: [https://github.com/rlgraph/rlgraph](https://github.com/rlgraph/rlgraph)

Documentation: Docs are available on [readthedocs](https://rlgraph.readthedocs.io/en/latest/?badge=latest)

Paper: Read our [paper](https://arxiv.org/abs/1810.09028) to learn more about RLgraph's design.     
