README

Pour lancer le projet :

-> ouvrir un terminal Windows depuis Anaconda

-> être à la racine du dossier "lsf discourd continu"
-> installer les requirements "pip install -r requirements.txt"
-> taper "python3 mLsfML3d.py"
-> écrire la commande indiquée :
	-> "python mLsfML3d.py --in ./select/ --mod ./model/ --set directory --nb 9999999999 --play sequence --aug no --naug 1 --dis hands --vec hands,pose --dim 3 --attempt 1 --sample 8 --xpol x --cfg medium --src lsf-xxxx --df lsf-xxxx_3_hp_8.npy --action infe > lsf-xxxx_3_hp_x_Flip_8.txt"
	-> remplacer les "xxxx" par "0064"
	-> remplacer "infe" par "demo" si demo souhaitée
-> lancer la commande