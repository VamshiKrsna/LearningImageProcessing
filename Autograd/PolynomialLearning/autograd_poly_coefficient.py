import torch 

# HW1
# trying out requires_grad 
x = torch.ones(1, requires_grad=True)
print(x) 
y = torch.sin(x) 
print(y) 
print(x.requires_grad) 
print(y.requires_grad)

x = torch.ones(1, requires_grad=True) 
y = x 
for i in range(10): 
    y = y * 2 + 1
print(y) 
print(y.backward()) 
print("Gradient of X  : ",x.grad) 
print("Gradient of Y  : ",y.grad) 
print("X requires grad  : ",x.requires_grad)
print("Y requires grad  : ",y.requires_grad) 

# HW2 Polynomial coefficient guesssing 

true_a = true_b = true_c = 1  # HIDDEN model doesn't know this 

x = torch.linspace(-3,3,300).unsqueeze(1)  # synthetic data generation (linearly spaced) 
y = true_a * x**2 + true_b * x + true_c  # ax2 + bx + c 

print("X : ", x) 
print("Y : ", y) 

# initialize results to 0.0 
pred_a = torch.tensor([0.0], requires_grad=True)
pred_b = torch.tensor([0.0], requires_grad=True)
pred_c = torch.tensor([0.0], requires_grad=True)
print(f"{pred_a},{pred_b},{pred_c}")

eta = 0.01 # learning rate 
for epoch in range(1001):  # 1000 epochs 
    y_pred = pred_a*x**2 + pred_b*x + pred_c  # updating predictions each iteration 
    loss = ((y_pred-y)**2).mean() # calculating Mean squared error MSE
    loss.backward() 
    with torch.no_grad(): 
        pred_a -= eta * pred_a.grad # learning rate into a,b,c 's gradient 
        pred_b -= eta * pred_b.grad 
        pred_c -= eta * pred_c.grad

        # pred_a.grad.zero_() # why reset  ?  This is not needed , takes by default 
        # pred_b.grad.zero_()
        # pred_c.grad.zero_()
        
    if epoch % 5 == 0: 
        print(epoch, pred_a.item(), pred_b.item(), pred_c.item(), loss.item()) 
