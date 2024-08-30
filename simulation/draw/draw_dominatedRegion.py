import numpy as np
import matplotlib.pyplot as plt

# Create a grid of points to evaluate
x = np.linspace(-2.5, 4, 400)
y = np.linspace(-2.5, 2.5, 300)
X, Y = np.meshgrid(x, y)

# Calculate distance conditions based on updated dominance definition
D = 1.25 * X**2 - 4.5 * X + 1.25 * Y**2 + 2.25  # The inequality from our derivation
D = 3 * X**2 - 8 * X + 3 * Y**2 + 4  # The inequality from our derivation
# Determine dominated region based on the derived condition
dominated_region = D < 0
# dominated_region = X > 0.5

# Plotting
plt.figure(figsize=(8, 6), facecolor='white')  # Set figure background to white
plt.scatter(0, 0, color='blue', label='Point A (0,0)', s=100)
plt.scatter(1, 0, color='red', label='Point B (1,0)', s=100)

# Fill dominated region
plt.contourf(X, Y, dominated_region, levels=1, colors=['white' ,'skyblue'], alpha=0.5)

# Set limits and labels
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')

plt.title('Dominated Region with Updated Definition (2 * d(B, C) < d(A, C))') #1.5 * 
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.show()
