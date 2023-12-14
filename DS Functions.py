import pandas as pd
football_data = {
    'Аты-жөні': ['Лионель Месси', 'Криштиану Роналду', 'Неймар', 'Мохамед Салах'],
    'Жасы': [36, 38, 31, 30],
    'Голдар саны': [672, 700, 308, 240]
}
df = pd.DataFrame(football_data)
oldest_player = df[df['Жасы'] == df['Жасы'].max()]['Аты-жөні'].values[0]
young_player = df[df['Жасы'] == df['Жасы'].min()]['Аты-жөні'].values[0]
average_goals = df['Голдар саны'].mean()
print("Ең егде футболшы: ", oldest_player)
print("Ең жас футболшы: ", young_player)
print("Голдардың орташа саны: ", average_goals)
