# for t in reversed(range(len(rewards))):
        #     # calculate G from the last transition
        #     running_return = self.gamma**0 * \
        #         rewards[t] + self.gamma * running_return
        #     returns[t] = running_return
        #     if returns.sum() == 0:
        #         vf[t] = 0.01
        #     else:
        #         vf[t] = running_return/returns.sum()
        #     # get value function estimates
        #     # advantage = returns[t] - vf[t]
        #     advantage = returns[t]

        #     # loss
        #     policies = net(first_embeddings, second_embeddings, states[t])
        #     # sum all features/embedding vectors of the state
        #     log_policies = (torch.log(policies) *
        #                     actions[t].detach()).sum(dim=1)
        #     loss = (-log_policies * advantage).sum()
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     total_loss += loss