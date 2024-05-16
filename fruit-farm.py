from environment import Environment

if __name__ == '__main__':
  environment = Environment(max_steps=20)
  
  environment.render()
  
  terminal = False
  while not terminal:
    terminal = environment.step()
    environment.render()