#  This is just an example.  You should modify this to suit your needs.

import Project2_agent as ag2
import Project2_env as env2
# import Project2_tests as tst


def main():

    wins: int = 0

    environment = env2.Blackjack()
    agent = ag2.MCAgent()
    agent.reset()
    # Check that the environment parameters match
    if (environment.get_number_of_states() == agent.get_number_of_states()) and \
            (environment.get_number_of_actions() == agent.get_number_of_actions()):

        # Play 500,000 games
        for i in range(0, 501):

            # reset the game and observe the current state
            current_state = environment.reset()
            game_end = False

            # Do until the game ends:
            agent.reset_trajectory()
            agent.reset()

            while True:
                action = agent.select_action(current_state)
                next_state, reward, game_end = environment.execute_action(action)
                agent.current_value = environment.agent_total
                print(f"\n\treward: {reward}\tcurrent value: {agent.current_value}")
                agent.store_trajectory(current_state, action, reward, next_state)
                agent.update_q(next_state, reward)
                current_state = next_state
                if game_end:
                    break

            # Episode over, so update the q table
            agent.update_q(next_state, reward)

            # --- Add data collection here as appropriate ---#
            if reward == 1:
                wins += 1

        print("\nProgram completed successfully.")
        print(f"win rate: {float(wins / 500)}")
    else:
        print("Environment and Agent parameters do not match. Terminating program.")


if __name__ == "__main__":
    main()
