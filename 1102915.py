import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path="C:\Windows\Fonts\msjh.ttc"
font_prop = FontProperties(fname=font_path)

# 定義 Ackley 函數
def ackley_function(x, y):
    a = 20 
    b = 0.2
    c = 2*np.pi
    sum_sq_term = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    cos_term = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return sum_sq_term + cos_term + a + np.exp(1)

# 初始化族群
def initialize_population(pop_size, bounds):
    x_population = np.random.uniform(bounds[0], bounds[1], pop_size)
    y_population = np.random.uniform(bounds[0], bounds[1], pop_size)
    print(x_population)
    print(y_population)
    return x_population, y_population

# 計算適應度
def calculate_fitness(x_population, y_population):
    fitness = np.array([ackley_function(x, y) for x, y in zip(x_population, y_population)])
    #print(f'適應度為：{fitness}')
    return fitness

# 交配(使用雙點交配)
def crossover(x_population, y_population):
    offspring_x = np.copy(x_population)
    offspring_y = np.copy(y_population)
    # 隨機選擇交配範圍
    idx1, idx2 = np.sort(np.random.randint(0, x_population.size, 2))
    #print(idx1,idx2)
    # 在範圍內進行交配
    tmp_x = np.copy(offspring_x)
    offspring_x[idx1:idx2] = offspring_y[idx1:idx2]
    offspring_y[idx1:idx2] =tmp_x[idx1:idx2]
    return offspring_x, offspring_y

# 變異
def inverse_mutation(offspring_x, offspring_y):
    mutation_x = np.copy(offspring_x)
    mutation_y = np.copy(offspring_y)
    # 隨機選擇變異範圍
    idx1, idx2 = np.sort(np.random.randint(0, offspring_y.size, 2))
    idx3, idx4 = np.sort(np.random.randint(0, offspring_y.size, 2))
    #print(idx1,idx2)
    # 在範圍內進行變異
    mutation_x[idx1:idx2] = [x/1.1 for x in mutation_x[idx1:idx2]]
    mutation_y[idx3:idx4] = [y/1.2 for y in mutation_y[idx3:idx4]]
    #為防止值被交換回原位
    return mutation_x,mutation_y

# 遺傳算法參數及初始化
pop_size = 100  # 族群大小
num_generations = 100  # 世代數
bounds = (-6, 6)  # x, y 的範圍
x_population, y_population = initialize_population(pop_size, bounds)

# 創建等高線圖的 Ackley 函數值
x = np.linspace(bounds[0], bounds[1], 400)
y = np.linspace(bounds[0], bounds[1], 400)
X, Y = np.meshgrid(x, y)
Z = ackley_function(X, Y)

# 初始化等高線圖
contour_fig, contour_ax = plt.subplots(figsize=(7, 5.6))
contour = contour_ax.contourf(X, Y, Z, levels=50, cmap='viridis')
cbar = contour_fig.colorbar(contour)
cbar.set_label('適應度', fontproperties=font_prop)
contour_ax.set_xlabel('x')
contour_ax.set_ylabel('y')
contour_ax.set_title('Ackley Function在演化計算的個體分布', fontproperties=font_prop)
# 調整視窗位置 - 這裡是示例，您可能需要根據您的屏幕分辨率調整位置
contour_fig_manager = plt.get_current_fig_manager()
contour_fig_manager.window.wm_geometry("+0+0")


# 初始化最佳適應度追踪圖形視窗
fitness_fig = plt.figure(figsize=(5.6, 5.6))
fitness_ax = fitness_fig.add_subplot(1, 1, 1)
fitness_ax.set_xlabel("回合" , fontproperties=font_prop)
fitness_ax.set_ylabel("適應度" , fontproperties=font_prop)
fitness_ax.set_title("每回合最佳適應度" , fontproperties=font_prop)
# 調整視窗位置
fitness_fig_manager = plt.get_current_fig_manager()
fitness_fig_manager.window.wm_geometry("+730+0")

# 進化過程和可視化
best_fitness_history = []  # 用於存儲每一代的最佳適應度
# 用來存儲散點圖對象的變量
scatter_plots = []
for generation in range(num_generations):
    offspring_x, offspring_y = crossover(x_population, y_population)
    x_population, y_population = inverse_mutation(offspring_x, offspring_y)
    fitness = calculate_fitness(x_population, y_population)
    best_fitness = np.min(fitness)
    best_idx = np.argmin(fitness)
    best_x = x_population[best_idx]
    best_y = y_population[best_idx]
    best_fitness_history.append(best_fitness)
    print(f"第{generation+1}回合","最佳個體:", (best_x, best_y), "最佳適應度:", best_fitness)

    # 更新等高線圖上的族群分布
    plt.figure(1)  # 切換到等高線圖視窗
    
    if scatter_plots:
            for sp in scatter_plots:
                sp.remove()
            scatter_plots.clear()
    sp_pop = contour_ax.scatter(x_population, y_population, color='red', s=10)
    sp_best = contour_ax.scatter(best_x, best_y, color='blue', s=30, edgecolors='white')  # 最佳個體
    scatter_plots.extend([sp_pop, sp_best])  # 添加新的散點圖到列表
    contour_fig.canvas.draw_idle()  # 重繪圖形但不阻塞
    plt.pause(0.01)

    # 更新最佳適應度追踪圖形
    fitness_ax.clear()
    fitness_ax.set_xlabel("回合", fontproperties=font_prop)
    fitness_ax.set_ylabel("適應度", fontproperties=font_prop)
    fitness_ax.set_title("每回合最佳適應度", fontproperties=font_prop)
    fitness_ax.plot(best_fitness_history, label='Best Fitness')
    fitness_ax.legend()
    fitness_fig.canvas.draw()
    plt.pause(0.01)

# 最終評估
print("最終結果: 最佳個體:", (best_x, best_y), "最佳適應度:", best_fitness)
plt.show()

