import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, input_dim, output_dim):
        self.model = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.memory = []
        self.gamma = 0.99

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < 0.1:  # epsilon-greedy policy
            return random.randint(0, 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < 1000:
            return
        batch = random.sample(self.memory, 64)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.model(next_states)

        target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)
        expected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.criterion(expected_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

agent = Agent(input_dim=10, output_dim=2)
#integrating 
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import pipeline

class ActionHandleQuery(Action):
    def name(self):
        return "action_handle_query"

    def __init__(self):
        self.classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message['text']
        intent = self.classifier(user_message)[0]['label'].lower()

        if intent == 'ask_order_status':
            dispatcher.utter_message(template="utter_ask_order_status")
        elif intent == 'ask_return_policy':
            dispatcher.utter_message(template="utter_return_policy")
        else:
            dispatcher.utter_message(template="utter_greet")

        return []
