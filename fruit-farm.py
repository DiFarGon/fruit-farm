from environment import Environment

if __name__ == '__main__':
  environment = Environment()
  
  environment.render()
  
  terminal = False
  while not terminal:
    terminal = environment.step()
    environment.render()