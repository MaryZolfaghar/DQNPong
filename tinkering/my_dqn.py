def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    """Run Deep Q-learning algorithm.
    You can specify your own convnet using q_func.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    ###############
    # BUILD MODEL #
    ###############

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph, tf.float32)/255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32)/255.0




    # q function and target q function

    prediction = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)

    target_prediction = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)


    # greedy exploration
    predicted_action = tf.argmax(prediction, 1)


    # error
    y = tf.cond(
        tf.equal(done_mask_ph[0], 1.0),
        lambda: rew_t_ph, lambda: tf.add(rew_t_ph, tf.multiply(gamma, tf.reduce_max(target_prediction, 1))))


    total_error = tf.reduce_mean(tf.square(y - tf.reduce_max(prediction, 1)))



def run(env, q_func, session, model_path,
    replay_buffer_size=1000000, frame_history_len=4, max_episode_count=500,
    render=False):
    """Run Deep Q-learning algorithm.
    You can specify your own convnet using q_func.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    session: tf.Session
        tensorflow session to use.
    model_path: path to the trained model
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    frame_history_len: int
        How many past frames to include as input to the model.
    max_episode_count: int
        maximum number of episodes to run
    """


    # TODO:
    # ReplayBuffer
    ## sotre_frames
    ## encode_recent_observation
    ## store_effect

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph, tf.float32)/255.0
    prediction = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)


    # greedy exploration
    predicted_action = tf.argmax(prediction, 1)


    # construct the replay buffer
    # replay_buffer_size=1000000, frame_history_len=4,
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)


     ###############
    # RUN ENV     #
    last_obs = env.reset()

    episode_count = 0


     if episode_count > max_episode_count:
                break


     idx = replay_buffer.store_frame(last_obs)


     action = session.run(
                predicted_action,
                feed_dict={
                    obs_t_ph:replay_buffer.encode_recent_observation().reshape(1, 84, 84, 4)
                }
            )[0]


    # exploration
    if random.random() < 0.01:
                    action = env.action_space.sample()



    obs, reward, done, _ = env.step(action)


    replay_buffer.store_effect(idx, action, reward, done)



    if done:
        obs = env.reset()
        episode_count += 1



    if render:
        env.render()




    last_obs = obs
