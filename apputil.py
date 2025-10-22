from collections import defaultdict
import numpy as np

class MarkovText(object):

    def __init__(self, corpus, k):
        self.corpus = corpus
        self.k = int(k)
        if self.k < 1:
            raise ValueError("k must be >= 1")
        # very light tokenization: split on whitespace
        self.tokens = self.corpus.split()
        if len(self.tokens) < self.k + 1:
            raise ValueError("Corpus too small for chosen k")
        self.term_dict = None  # you'll need to build this

    def get_term_dict(self):

        """
        Build a transition dictionary mapping a state -> list of possible next tokens.
        For k==1, state is a single token string.
        For k>1, state is a tuple of k tokens.
        """
        trans = defaultdict(list)
        toks = self.tokens
        k = self.k

        # iterate states and push following token
        for i in range(len(toks) - k):
            state = toks[i] if k == 1 else tuple(toks[i:i+k])
            nxt = toks[i + k]
            trans[state].append(nxt)

        # save
        self.term_dict = dict(trans)
        return self.term_dict


    def generate(self, seed=None, term_count=15):
        """
        Generate text using the transition dictionary.
        - term_count: number of tokens to produce
        - seed: optional starting token/tuple. If provided and not found, raise ValueError.
        Rules:
        - At each step choose the next token uniformly from the state's list.
        - If a state has no outgoing transitions (rare with this build), we re-seed randomly.
        """
        if self.term_dict is None:
            self.get_term_dict()

        k = self.k
        # Normalize/validate seed
        if seed is not None:
            if k == 1:
                state = seed
            else:
                # seed can be string or tuple/list of length k
                if isinstance(seed, str):
                    # allow string with spaces
                    parts = seed.split()
                else:
                    parts = list(seed)
                if len(parts) != k:
                    raise ValueError(f"Seed must have {k} tokens for k={k}")
                state = tuple(parts)
            if state not in self.term_dict:
                raise ValueError("Seed term not found in corpus/state space.")
        else:
            # pick a random valid starting state
            keys = list(self.term_dict.keys())
            state = keys[np.random.randint(0, len(keys))]

        # initialize output with state
        out = []
        if k == 1:
            out.append(state)
        else:
            out.extend(list(state))

        # generate
        for _ in range(max(0, term_count - (len(out)))):
            choices = self.term_dict.get(state, [])
            if not choices:
                # dead end: random re-seed
                keys = list(self.term_dict.keys())
                state = keys[np.random.randint(0, len(keys))]
                if k == 1:
                    out.append(state)
                else:
                    out.extend(list(state))
                continue

            nxt = np.random.choice(choices)
            out.append(nxt)

            # roll the state
            if k == 1:
                state = nxt
            else:
                state = tuple(list(state)[1:] + [nxt])

        return " ".join(out)