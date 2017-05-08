# Eye Tracker
Implemented and improved the iTracker model proposed in paper [Eye Tracking for Everyone](https://arxiv.org/abs/1606.05814).

![](itracker_arch.png)
*<center><h3>Figure 1: itracker architecture</h3></center>*

![](itracker_adv_arch.png)
*<center><h3>Figure 2: modified itracker architecture</h3></center>*

Figure 1 and 2 show the architectures of the iTracker model
and the modified model. The only difference between the modified model and the iTracker model is
that we concatenate the face layer FC-F1 and face mask layer FC-FG1 first, after applying a fully connected layer FC-F2,
we then concatenate the eye layer FC-E1 and FC-F2 layer.
We claim that this modified architecture is superior to the iTracker architecture.
Intuitively, concatenating the face mask information together with the eye information
may confuse the model since the face mask information is irrelevant to the eye information.
Even though the iTracker model succeeded to learn this fact from the data,
the modified model outperforms the iTracker model by explictlying encoded with this knowledge.
In experiments, the modified model converged faster (28 epochs vs. 40+ epochs) and achieved better validation
error (2.19 cm vs. 2.514 cm).
The iTracker model was implemented in itracker.py and the modified one was
implemented in itracker_adv.py.
Note that a smaller dataset was used in experiments and no data augmentation was applied.
You can download the dataset [here]().
