forward_system_prompt = "You are a chemist. Your task is to predict the SMILES representation of the product molecule given a precursor."
forward_user_prompt_templates = [
    "[SMILES] Considering the given starting materials, what might be the resulting product in a chemical reaction?",
    "Consider that for a chemical reaction, if [SMILES] is/are the reactants and reagents, what can be the product?",
    "[SMILES] Given the above reactants and reagents, what could be a probable product of their reaction?",
    "Predict a possible product from the listed reactants and reagents. [SMILES]",
    "Using [SMILES] as the reactants and reagents, tell me the potential product.",
    "Please provide a feasible product that could be formed using these reactants and reagents: [SMILES] .",
    "A chemical reaction has started with the substance(s) [SMILES] as the reactants and reagents, what could be a probable product?",
    "Propose a potential product given these reactants and reagents. [SMILES]",
    "Can you tell me the potential product of a chemical reaction that uses [SMILES] as the reactants and reagents?",
    "Based on the given reactants and reagents: [SMILES], what product could potentially be produced?",
    "[SMILES] Based on the reactants and reagents given above, suggest a possible product.",
    "Given the following reactants and reagents, please provide a possible product. [SMILES]",
    "Predict the product of a chemical reaction with [SMILES] as the reactants and reagents.",
]

retro_system_prompt = "You are a chemist. Your task is to predict the SMILES representation of the reactant molecule given a product."
retro_user_prompt_templates = [
    "With the given product [SMILES], suggest some likely reactants that were used in its synthesis.",
    "[SMILES] Given the product provided, propose some possible reactants that could have been employed in its formation.",
    "Can you list the reactants that might result in the chemical product [SMILES] ?",
    "To synthesis [SMILES], what are the possible reactants? Write in the SMILES representation.",
    "Can you identify the reactant(s) that might result in the given product [SMILES] ?",
    "Do retrosynthesis with the product [SMILES] .",
    "Based on the given product, provide some plausible reactants that might have been utilized to prepare it. [SMILES]",
    "Suggest possible substances that may have been involved in the synthesis of the presented compound. [SMILES]",
    "Identify possible reactants that could have been used to create the specified product. [SMILES]",
    "Given the following product, please provide possible reactants. [SMILES]",
    "Provide the potential reactants that may be used to produce the product [SMILES] .",
    "Could you tell which reactants might have been used to generate the following product? [SMILES]",
    "What reactants could lead to the production of the following product? [SMILES]",
]

reagent_system_prompt = "You are a chemist. Now you are given a reaction equation. The reaction equation has the following format:\n```\nreactant1.reactant2. ... .reactantN>>product\n```\nYour task is to predict the SMILES representation of the reagents."
reagent_user_prompt_templates = [
    "Please suggest some possible reagents that could have been used in the following chemical reaction [SMILES].",
    "Given this chemical reaction [SMILES], what are some reagents that could have been used?",
    "Given the following reaction [SMILES], what are some possible reagents that could have been utilized?",
    "[SMILES] Based on the given chemical reaction, can you propose some likely reagents that might have been utilized?",
    "[SMILES] From the provided chemical reaction, propose some possible reagents that could have been used.",
    "Can you provide potential reagents for the following chemical reaction? [SMILES]",
    "[SMILES] Please propose potential reagents that might have been utilized in the provided chemical reaction.",
    "What reagents could have been utilized in the following chemical reaction? [SMILES]",
    "Based on the given chemical reaction [SMILES], suggest some possible reagents.",
    "Given the following chemical reaction [SMILES], what are some potential reagents that could have been employed?",
    "Please provide possible reagents based on the following chemical reaction [SMILES].",
    "Can you suggest some reagents that might have been used in the given chemical reaction? [SMILES]",
]
reagent_negative_prompt_templates = [
    "No additional reagents are needed for this reaction.",
    "The transformation proceeds without any external reagent.",
    "This conversion is reagent-free.",
    "The reaction occurs in the absence of added reagents.",
    "No chemical reagent is required to drive the process.",
    # "The substrates react autonomously, needing no extra reagents.",
    # "The pathway completes without introducing a reagent.",
    "No supplementary reagents participate in this step.",
    # "This is an inherently reagent-less reaction.",
    # "This transformation proceeds without any external reagents.",
    # "No supplementary chemicals are required—the substrates react on their own.",
    # "The reaction is reagent-free, relying solely on intrinsic functional groups.",
    # "Nothing else needs to be added; the process is self-sufficient.",
    "No additional reagents are involved in carrying out this conversion.",
    # "The system advances spontaneously, so external reagents are unnecessary.",
    # "Because the substrates contain all needed functionality, no reagent is introduced.",
    # "The reaction operates under neat conditions, with zero reagents beyond the starting materials.",
    # "No further chemical input is needed; the transformation is achieved reagent-less.",
    # "This is an intramolecular process, requiring no extra reagents to proceed.",
]

catalyst_system_prompt = "You are a chemist. Now you are given a reaction equation. The reaction equation has the following format:\n```\nreactant1.reactant2. ... .reactantN>>product\n```\nYour task is to predict the SMILES representation of the catalyst."
catalyst_user_prompt_templates = [
    "Can you suggest some catalysts that might have been used in the given chemical reaction? [SMILES]",
    "What catalysts could have been utilized in the following chemical reaction? [SMILES]",
    "Given this chemical reaction [SMILES], what are some catalysts that could have been used?",
    "Please provide possible catalysts based on the following chemical reaction [SMILES].",
    "[SMILES] From the provided chemical reaction, propose some possible catalysts that could have been used.",
    "Please suggest some possible catalysts that could have been used in the following chemical reaction [SMILES].",
    "[SMILES] Please propose potential catalysts that might have been utilized in the provided chemical reaction.",
    "Can you provide potential catalyst for the following chemical reaction? [SMILES]",
    "Given the following reaction [SMILES], what are some possible catalysts that could have been utilized?",
    "[SMILES] Based on the given chemical reaction, can you propose some likely catalysts that might have been utilized?",
    "Based on the given chemical reaction [SMILES], suggest some possible catalyst.",
    "Given the following chemical reaction [SMILES], what are some potential catalysts that could have been employed?",
]

solvent_system_prompt = "You are a chemist. Now you are given a reaction equation. The reaction equation has the following format:\n```\nreactant1.reactant2. ... .reactantN>>product\n```\nYour task is to predict the SMILES representation of the solvents."
solvent_user_prompt_templates = [
    "[SMILES] Please propose potential solvents that might have been utilized in the provided chemical reaction.",
    "Can you suggest some solvents that might have been used in the given chemical reaction? [SMILES]",
    "Based on the given chemical reaction [SMILES], suggest some possible solvents.",
    "[SMILES] From the provided chemical reaction, propose some possible solvents that could have been used.",
    "Given the following chemical reaction [SMILES], what are some potential solvents that could have been employed?",
    "Please provide possible solvents based on the following chemical reaction [SMILES].",
    "[SMILES] Based on the given chemical reaction, can you propose some likely solvents that might have been utilized?",
    "Given the following reaction [SMILES], what are some possible solvents that could have been utilized?",
    "Can you provide potential solvents for the following chemical reaction? [SMILES]",
    "What solvents could have been utilized in the following chemical reaction? [SMILES]",
    "Given this chemical reaction [SMILES], what are some solvents that could have been used?",
    "Please suggest some possible solvents that could have been used in the following chemical reaction [SMILES].",
]
solvent_negative_prompt_templates = [
    "No solvent is needed for this reaction.",
    "The transformation proceeds without any added solvent.",
    "This conversion is solvent-free.",
    "The reaction occurs in the absence of a solvent phase.",
    "No liquid medium is required to drive the process.",
    "The substrates react neat, with no solvent addition.",
    "The pathway completes without introducing a solvent.",
    "No auxiliary solvent participates in this step.",
    # "This is an inherently solvent-less reaction.",
    "This reaction proceeds without any solvent.",
    # "No medium is added—the transformation is performed under neat conditions.",
    # "The process is entirely solvent-free, relying on the substrates alone.",
    # "No external solvent is required; the components react as-is.",
    # "The conversion is carried out neat, with zero solvent involvement.",
    # "Because the substrates can mix and react directly, a solvent is unnecessary.",
    # "This operation runs in the absence of solvent, eliminating dilution altogether.",
    # "The system advances solvent-less, maximizing atom economy.",
    "No liquid medium is introduced; the reaction occurs in its own melt.",
    "A dedicated solvent is not employed—the chemistry proceeds in a solvent-free environment.",
]
pretraining_system_prompt = "You are a chemist."