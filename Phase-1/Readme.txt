-- code contains all the source code files
        -- ColorMoments.py : Code for CM feature description calculation and matching score
        -- SIFT.py : Code for SIFT feature description calculation and matching score
        -- create_feature_script.py : Script to perform task1, task2, task3
-- output
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


