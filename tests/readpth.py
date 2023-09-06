import torch  
 
checkpoint = torch.load("experiments/NAFNetL1Loss_pickbands/models/net_g_50000.pth")
#torch.load('路径') 但是我的电脑没有GPU，是集成显卡呜呜呜，所以还得加个后面那部分map_location=torch.device('cpu')
 
print(checkpoint['params'].keys())   
 
 
print(checkpoint['params'].shape)