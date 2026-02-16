import os
import AttackMethods

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Main method to do one of the ResNet20  experiments 
#Uncomment any one of the following lines to run an attack 
def main():
    # -----------------------------
    # Default: APGD-L2 on ResNet20
    # -----------------------------
	  AttackMethods.APGD_L2_ResNet20()

    # -----------------------------
    # Other attacks
    # -----------------------------
    # AttackMethods.APGD_Linf_ResNet20()
    # AttackMethods.L0_PGD_ResNet20()
    # AttackMethods.L0_Sigma_PGD_ResNet20()
    # AttackMethods.L0_Linf_PGD_ResNet20()
    # AttackMethods.APGD_L1_ResNet20()


if __name__ == "__main__":
    main()
