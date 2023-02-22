#  This is just an example.  You should modify this to suit your needs.

import Project2_agent as ag2
import Project2_env as env2
# import Project2_tests as tst
import matplotlib.pyplot as plt


def plot_data(rewards_list, window:int=1000, use_window:bool=False):
    """
    Helper function for plotting results of Monte Carlo BlackJack
    :param rewards_list: list - the sequence of rewards from each of the m agents
    :param window: int - a moving average window for smoothing learning curves
    :param use_window: bool - a flag indicating whether or not thw window should be used
    :return: null
    """
    x_list: list = []
    y_list: list = []

    for rewards in rewards_list:
        x: list = list(range(0, len(rewards)))
        y: list = []
        win_count: int = 0
        for i in range(len(rewards)):
            if rewards[i] == 1:
                win_count += 1
            y.append(float(win_count / (i + 1)))

        if use_window:
            y_final: list = []
            win_count_ma: int = 0
            for i in range(len(rewards)):
                if rewards[i] == 1:
                    win_count += 1
                    win_count_ma += 1
                y.append(float(win_count / (i + 1)))
                if i % window == 0 and i > 10:
                    y_final.append(float(win_count_ma / window))
                    win_count_ma = 0
            x = list(range(0, len(y_final)))
            y = y_final

        x_list.append(x)
        y_list.append(y)

    for i in range(len(x_list)):
        plt.plot(x_list[i], y_list[i], label=f"agent {i + 1}")
    if use_window:
        plt.title(f"# Trials vs % Wins with Window = {window}")
    else:
        plt.title("# Trials vs % Wins")
    plt.xlabel('# Trials')
    plt.ylabel('% Wins')
    plt.legend(loc='upper right')
    plt.show()


def main(should_plot:bool=True):
    """
    Main function that sets up Monte Carlo analysis for Black Jack environment and trains
     m agents over n episodes.
    :param should_plot: bool - flag indicating whether or not to plot results
    :return: null
    """
    ma_window: int = 1000
    episodes: int = 500001
    all_agents_wins: list = []

    for player in range(10):
        wins_all: int = 0
        environment = env2.Blackjack()
        agent = ag2.MCAgent()
        agent.reset()
        all_rewards: list = []

        # Check that the environment parameters match
        if (environment.get_number_of_states() == agent.get_number_of_states()) and \
                (environment.get_number_of_actions() == agent.get_number_of_actions()):

            # Play 500,000 games
            for i in range(0, episodes):

                # reset the game and observe the current state
                current_state = environment.reset()

                # Do until the game ends:
                agent.reset_trajectory()
                agent.reset()

                while True:
                    action = agent.select_action(current_state)
                    next_state, reward, game_end = environment.execute_action(action)
                    agent.current_value = environment.agent_total
                    agent.store_trajectory(current_state, action, reward, next_state)
                    agent.update_q(next_state, reward)
                    current_state = next_state
                    if game_end:
                        break

                # Episode over, so update the q table
                agent.update_q(next_state, reward)

                # --- Add data collection here as appropriate ---#
                if reward == 1:
                    wins_all += 1
                all_rewards.append(reward)

            all_agents_wins.append(all_rewards)
            print(f"win rate: {float(wins_all / episodes)}")
        else:
            print("Environment and Agent parameters do not match. Terminating program.")

    if should_plot:
        plot_data(all_agents_wins)
        plot_data(all_agents_wins, ma_window, True)



if __name__ == "__main__":
    main(should_plot=True)


"""

Program completed successfully.
win rate: 0.44081311837376325

Program completed successfully.
win rate: 0.44023311953376093

Program completed successfully.
win rate: 0.4401471197057606

Program completed successfully.
win rate: 0.4396511206977586

Program completed successfully.
win rate: 0.44001511996976006

Program completed successfully.
win rate: 0.4392811214377571

Program completed successfully.
win rate: 0.4407791184417631

Program completed successfully.
win rate: 0.44048511902976195

Program completed successfully.
win rate: 0.4416451167097666

Program completed successfully.
win rate: 0.44009311981376037
"""
