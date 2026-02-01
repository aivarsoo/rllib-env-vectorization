# Attemps to vectorize environments in Ray RL-lib

In this repo, we are exploring a couple of ways to vectorize envrionments in Ray [RL-lib library](https://github.com/ray-project/ray/tree/master/rllib)

we are focusing on the case of `vector_entry_point`, where the user is vectorizing the environment themselves and passes it to RL-lib. 

There are two choices at the moment for vectorization

`vec` - the indended way of vectorizing the environment 

`packed_vec` - a proposed way to vectorize the environment. 
