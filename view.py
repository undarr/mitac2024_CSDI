import torch
import matplotlib.pyplot as plt
import numpy as np

def drift(St, foldername):
    LSt=torch.log(torch.maximum(St, torch.tensor(0.001)))
    LdifSt=LSt[1:,::]-LSt[:-1,::]
    LdifStSq=LdifSt**2
    print(LdifSt.size())
    T=1
    dt=torch.sum(LdifSt, axis=0)/T
    print(dt[:10])
    print(dt.size(),torch.mean(dt),torch.median(dt),torch.var(dt))
    dt=dt.numpy()
    
    # Calculate the PDF
    pdf, bins = np.histogram(dt, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot the PDF
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, pdf, color='b')
    plt.axvline(x=np.mean(dt), color='b', linestyle='--', linewidth=2)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title("Prediction volatility")
    plt.savefig(foldername+"/zz_prediction_drift.png")
    plt.close()

def volatility(St, foldername):
    T=1
    LSt=torch.log(torch.maximum(St, torch.tensor(0.001)))
    LdifSt=LSt[1:,::]-LSt[:-1,::]
    LdifStSq=LdifSt**2
    rvt=torch.sum(LdifStSq, axis=0)
    rvtt=(rvt/T)**0.5
    print(rvtt[:10])
    print(torch.mean(rvtt),torch.median(rvtt),torch.var(rvtt))
    rvtt=rvtt.numpy()

    # Calculate the PDF
    pdf, bins = np.histogram(rvtt, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot the PDF
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, pdf, color='b')
    plt.axvline(x=np.mean(rvtt), color='b', linestyle='--', linewidth=2)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title("Prediction volatility")
    plt.savefig(foldername+"/zz_prediction_volatility.png")
    plt.close()

def viewresult(foldername, i=0, ds=1):
    generation_result_BxGenCxTS = torch.load(foldername+'samples'+str(ds)+'.pt', map_location=torch.device('cpu')) #prediction
    time_vector_BxTS = torch.load(foldername+'observed_time.pt', map_location=torch.device('cpu')) #time
    testset_ground_truth_BxTS = torch.load(foldername+'c_target'+str(ds)+'.pt', map_location=torch.device('cpu')) #original
    testset_imputation_map_BxTS = torch.load(foldername+'eval_points.pt', map_location=torch.device('cpu'))
    trainingdata_traincountxTS = torch.load(foldername+'traindata.pt', map_location=torch.device('cpu'))

    batch_size=generation_result_BxGenCxTS.size()[0]
    gencount=generation_result_BxGenCxTS.size()[1]
    traincount=trainingdata_traincountxTS.size()[0]
    ts=generation_result_BxGenCxTS.size()[2]
    ground_truth_duplicated_BxTS = testset_ground_truth_BxTS[i].repeat(gencount,1)

    tt_gen = time_vector_BxTS[i].repeat(gencount,1).T
    tt_train = time_vector_BxTS[i].repeat(traincount,1).T
    St=((1-testset_imputation_map_BxTS[i])*testset_ground_truth_BxTS[i]+testset_imputation_map_BxTS[i]*generation_result_BxGenCxTS[i]).T

    plt.plot(tt_gen, St)
    plt.xlabel("Time steps")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title("Predictions Full")
    plt.savefig(foldername+"/zz_prediction.png")
    plt.close()

    print(trainingdata_traincountxTS.T.size())
    plt.plot(tt_train, trainingdata_traincountxTS.T)
    plt.xlabel("Time steps")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title("Training data")
    plt.savefig(foldername+"/zz_training_data_plot.png")
    plt.close()

    drift(St, foldername)
    volatility(St, foldername)
