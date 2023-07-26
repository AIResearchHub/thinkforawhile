# Transformer Gallery

Think-for-a-while is a new idea that aims to teach recurrent transformers to self improve via unrolling. Since different unroll lengths are good for different tasks, we seek to measure a confidence score associated with each prediction and use it as target for less confident unroll lengths. This approach could build knowledge upon existing datasets like mathematical reasoning where some mundane tasks such as 3 digit multiplication requires many small multiplications added together, which should be enhanced with multiple passes through the network. The model can then arrive at better conclusions using building blocks from its own knowledge.

## Recurrent Transformers

### Block Recurrent Transformer

Block Recurrent Transformer proposes a recurrent layer that maintains a recurrent state like a RNN. In the recurrence layer, 2 self attention and 2 cross attention is performed with 4 different sets of queries from the inputs and recurrent state. BPTT is performed inside the sequence by dividing the it into windows.

### Recurrent Memory Transformer

Recurrent Memory Transformer uses memory tokens to pass on information to the next recurrent time step where gradients are flowed backwards during BPTT. At each timestep, the memory tokens are concatenated from left and right, and new memory tokens are written to the right of the output tokens.

## Transformer Base Code

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=AIResearchHub&repo=transformergallery)](https://github.com/AIResearchHub/transformergallery)

## Citations

