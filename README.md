Load the ipynb into Colab, make sure the runtime is T4, not CPU.

Load data from WhatsApp into Colab.
Run the notebook, it'll only take a couple of seconds.
Understand all the code using NotebookLM and base paper is called "UrbanGraph.pdf" in root.
Reference the architecture with "UHI_Kolkata.pdf" in root.

Everything else is done in terms of data loading, the model just needs to converge, make it work. I'll work on the paper after that. The RSME and MAE results are misleading, so don't worry about that. Just make sure the model converges and the code runs without errors.

# ISSUES FIXED (Adi , 14/4/2026)

- GNN Collapse Fixed: Your implementation of PairNorm, F.leaky_relu, and the residual connection (out = out + self_emb) has successfully diversified the node embeddings. The mean cosine similarity dropped from 0.9925 to 0.0119, meaning the nodes are now distinct.
- Training and Convergence: The model is training without errors. The MSE loss decreased from 1.5840 to 0.0763 over 40 epochs, showing clear convergence.
- Code Execution: The pipeline from data preparation to hybrid temporal-spatial prediction is running smoothly on the GPU.

The code is now structurally sound and the model is learning meaningful spatial features. Since you noted that RMSE/MAE are currently misleading due to the dummy data or specific dataset scaling, the focus remains on this successful architectural convergence.
