# Human Hand Search and Classification System
 This repository consists of the project regarding CSE 515 MultiMedia and Web databases course. The project was divided into 3 phases. The details about the goals of each phase and how to run the code of each phase has been mentioned below:
 
 Phase 1:
 	The goal of phase 1 of the project is to get familiar with the 11k hand's image dataset and understand the concept of feature extraction and implementation of the two image feature extraction models. The two implemented models are Scale-invariant feature transformation and Color Moments, which helped in understanding the feature vector space/ feature descriptor of each implemented model. Once the features are extracted for the images, these features are used to retrieve the k similar images from the database for the given image. To find k similar images, the similarity/ distance functions are implemented, and the results obtained using these functions showed the importance of each function and characteristics of the images it captures.
	
		-- code contains all the source code files
			-- ColorMoments.py : Code for CM feature description calculation and matching score
			-- SIFT.py : Code for SIFT feature description calculation and matching score
			-- create_feature_script.py : Script to perform task1, task2, task3

			-- CM : Contains all the descriptor output in csv format
			    -- match : contains last k matched images for task 3
			-- SIFT : Contains all the descriptor output in csv format
			    -- match : contains last k matched images for task 3

		Example query to execute task 1 :-
		    python create_feature_script.py -m SIFT -d Images -i Hand_0009445.jpg -s

		Example query to execute task 2 :-
		    python create_feature_script.py -m SIFT -d Images

		Example query to execute task 3 :-
		    python create_feature_script.py -m SIFT -d Images -i Hand_0009445.jpg -k 5 -r

	Note:- Keep code, <input image directory> and output at same level
		
		

Phase 2:
	The project aims at developing a strong foundation in handling and processing high dimensional data and experiment with  various dimensionality reduction techniques like PCA, SVD, NMF and LDA and similarity and distance measures like euclidean distance and cosine similarity. The dataset used in this project is associated with the publication “Mahmoud Afifi. “11K Hands: Gender recognition and biometric identification using a large dataset of hand images.” M. Multimed Tools Appl (2019) 78: 20835.” The dataset has been downloaded from the following website. https://sites.google.com/view/11khands 
	
		-- code contains all the source code files
				-- phase2_main_script.py : Script to perform all the tasks
				-- ColorMoments.py : Code for CM feature description calculation and matching score
				-- SIFT.py : Code for SIFT feature description calculation and matching score
				-- HOGmain.py: Code for HOG feature description calculation and matching score
				-- LocalBinaryPatterns.py: Code for LBP feature description calculation and matching score
				-- BOW_compute.py: Computes bag of words for run-time image
				-- feature_descriptor.py: Stores feature descriptors and bag of words in DB
				-- LatentDirichletAllocation.py: Handles all the tasks related to LDA
				-- PrincipleComponentAnalysis.py: Handles all the tasks related to PCA
				-- SingularValueDecomposition.py: Handles all the tasks related to SVD
				-- NonNegativeMatrix.py: Handles all the tasks related to NMF
				-- SimilarSubject.py: Handles subject specific tasks
				-- Visualizer.py: Handles the visualizations of the project


		-- csv  -- Contains csv files metadata

		Initial Set-up:-
			python phase2_main_script.py -d ../Dataset2 -t 0
			mongoimport --port 27018 --db imagedb --type csv --file Desktop/Study/MWDB/ProjectTest/csv/ImageMetadata.csv --headerline
			python phase2_main_script.py -t 9

		Example query to execute task 1 :-
			python phase2_main_script.py -M CM -k 20 -T LDA -t 1

		Example query to execute task 2 :-
			python phase2_main_script.py -k 10 -m 10 -i Hand_0000111.jpg -T PCA -M HOG -t 2

		Example query to execute task 3 :-
			python phase2_main_script.py -k 20 -l left -T LDA -M LBP -t 3

		Example query to execute task 4 :-
			python phase2_main_script.py -k 10 -m 10 -i Hand_0000200.jpg -l palmar -T NMF -M LBP -t 4

		Example query to execute task 5 :-
			python phase2_main_script.py -k 10 -d ../phase2 -i Hand_0000896.jpg -l right -T SVD -M SIFT -t 5

		Example query to execute task 6 :-
			python phase2_main_script.py -s 27 -t 6

		Example query to execute task 7 :-
			python phase2_main_script.py -k 10 -t 7

		Example query to execute task 8 :-
			python phase2_main_script.py -k 4 -t 8

		Note:- Keep code, <input image directory> and csv at same level
		
		

Phase 3:
	The third phase of the project focuses on interpreting and analyzing data that can support efficient indexing, clustering, classification and relevance feedback system of the data items. In this phase of the project, we are using Support Vector Machine (SVM), K Means Clustering, Decision Tree, Personalized Page Rank (PPR), Relevance Feedback System and Locality Sensitive Hashing. The various algorithms used are evaluated and visualized for each task. The dataset used in this project is associated with the publication “Mahmoud Afifi. “11K Hands: Gender recognition and biometric identification using a large dataset of hand images.” M. Multimed Tools Appl (2019) 78: 20835.” 

		-- code contains all the source code files
				-- phase3_main_script.py : Script to perform all the tasks
				-- ColorMoments.py : Code for CM feature description calculation and matching score
				-- SIFT.py : Code for SIFT feature description calculation and matching score
				-- HOGmain.py: Code for HOG feature description calculation and matching score
				-- LocalBinaryPatterns.py: Code for LBP feature description calculation and matching score
				-- BOW_compute.py: Computes bag of words for run-time image
				-- feature_descriptor.py: Stores feature descriptors and bag of words in DB
				-- LatentDirichletAllocation.py: Handles all the tasks related to LDA
				-- PrincipleComponentAnalysis.py: Handles all the tasks related to PCA
				-- SingularValueDecomposition.py: Handles all the tasks related to SVD
				-- NonNegativeMatrix.py: Handles all the tasks related to NMF
				-- SimilarSubject.py: Handles subject specific tasks
				-- Visualizer.py: Handles the visualizations of the project


		-- csv  -- Contains csv files metadata

		Initial Set-up:-
			python phase3_main_script.py -d ../Dataset -t 0

		Example query to execute task 1 :-
			python phase3_main_script.py -t 1 -k =30 -l labelled_set1 -u unlabelled_set2

		Example query to execute task 2 :-
			python phase3_main_script.py -t 2 -c 5 -l labelled_set2 -u unlabelled_set1

		Example query to execute task 3 :-
			python phase3_main_script.py -t 3 -c 5 -m 10 -I "Hand_0008333.jpg Hand_0006183.jpg Hand_0000074" -l labelled_set2

		Example query to execute task 4 :-
			python phase3_main_script.py -t 4 -c 5 -T PPR -l labelled_set2 -u unlabelled_set2

		Example query to execute task 5 :-
			python phase3_main_script.py -t 5 -i Hand_0000674.jpg -m 20 -L 10 -k 10


		Note:- Keep code, <input image directory> and csv in the same folder
