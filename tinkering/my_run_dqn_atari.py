def get_env(task, seed, logs_path=None):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = logs_path or OPENAI_LOGS #'/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True, video_callable=False)
    env = wrap_deepmind(env)

    return env






def atari_run(env, session, model_path, max_episode_count):
    dqn.run(
        env,
        q_func=atari_model,
        session=session,
        model_path=model_path,
        replay_buffer_size=1000000,
        frame_history_len=4,
        max_episode_count=max_episode_count
    )

    env.close()




def run(env, q_func, session, model_path,
    replay_buffer_size=1000000, frame_history_len=4, max_episode_count=500,
    render=False):
    """Run Deep Q-learning algorithm.
