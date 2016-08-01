# The Wikipedia Bob Alice HMM example using hmmlearn
# Based on <http://sujitpal.blogspot.se/2013/03/the-wikipedia-bob-alice-hmm-example.html> and <https://en.wikipedia.org/wiki/Hidden_Markov_model#A_concrete_example>

import numpy as np
from hmmlearn import hmm #From <http://hmmlearn.readthedocs.io/>

np.random.seed(42)

states = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

start_probability = np.array([0.6, 0.4])

transition_probability = np.array([
  [0.7, 0.3],
  [0.4, 0.6]
])

emission_probability = np.array([
  [0.1, 0.4, 0.5],
  [0.6, 0.3, 0.1]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_  = transition_probability
model.emissionprob_ = emission_probability

# predict a sequence of hidden states based on visible states
bob_says = np.array([[0], [2], [1], [1], [2], [0]])
logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
print("Bob says:    %s"%", ".join([observations[x[0]] for x in bob_says]))
print("Alice hears: %s"%", ".join([states[x] for x in alice_hears]))
