from environment import Environment

if __name__ == '__main__':
  environment = Environment(grid_shape=(10, 10))
  
  while True:
    environment.render()