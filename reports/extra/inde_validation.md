# Indian English (`eng_Latn+inde`) validation

*Exploratory condition.* Method: Multi-VALUE (Ziems et al., 2023, ACL; https://aclanthology.org/2023.acl-long.44), `Dialects.IndianDialect` = eWAVE variety "Indian English" (IndE), applied to the 300 sampled Belebele English items (passage, question, 4 options). Package `value-nlp==0.3; spacy==3.7.4 en_core_web_sm==3.7.1; stanza==1.8.2; seed=1909 (per-field sha256 seed)`. No LLM is involved.

**Quality gate.** Multi-VALUE is rule-based over spaCy/Stanza parses and misfires on the short fragments that Belebele options are (for example "Meat" -> "Are Meating", "Police" -> "Polouses"). Every field whose LaBSE cosine to the original is below 0.85 is therefore reverted to the original English in the condition file; the unfiltered Multi-VALUE output of every field is kept in `mv_raw_fields` of each row. Statistics below are given for the raw output first, then for the final (gated) file.

## Raw Multi-VALUE output

- Rows: **300**; items with at least one changed field: **100.0%**
- Items flagged (any field LaBSE cosine < 0.85): **68** (22.7%); fields below 0.85: 89 of 1800
- Items where Stanza/spaCy sentence alignment failed on the whole field and Multi-VALUE was run sentence by sentence: 56
- Option collisions reverted: 0

| field | fraction changed | mean edit ratio (all) | mean edit ratio (changed only) | mean LaBSE cos | min LaBSE cos | n < 0.85 |
|---|---|---|---|---|---|---|
| flores_passage | 1.000 | 0.1525 | 0.1525 | 0.9850 | 0.9320 | 0 |
| question | 1.000 | 0.2383 | 0.2383 | 0.9485 | 0.6007 | 9 |
| mc_answer1 | 0.723 | 0.1946 | 0.2690 | 0.9500 | 0.5275 | 21 |
| mc_answer2 | 0.737 | 0.2032 | 0.2758 | 0.9474 | 0.3995 | 23 |
| mc_answer3 | 0.673 | 0.1997 | 0.2966 | 0.9526 | 0.5014 | 18 |
| mc_answer4 | 0.707 | 0.2079 | 0.2942 | 0.9569 | 0.7455 | 18 |
| **all fields** | 0.807 | 0.1994 | 0.2472 | 0.9567 | 0.3995 | 89 |

LaBSE = `sentence-transformers/LaBSE`, cosine between original and transformed field. Edit ratio = character Levenshtein distance / max length.

## Final file (after the gate)

- Items with at least one changed field: **100.0%** (300/300); fields changed: **75.7%**; fields reverted by the gate: 89; mean character edit ratio over all fields: 0.1792; all four options distinct in every row: True

| field | fraction changed (final) | fields reverted by gate |
|---|---|---|
| flores_passage | 1.000 | 0 |
| question | 0.970 | 9 |
| mc_answer1 | 0.653 | 21 |
| mc_answer2 | 0.660 | 23 |
| mc_answer3 | 0.613 | 18 |
| mc_answer4 | 0.647 | 18 |

**Known tool limitation.** The `progressives` rule also fires on sentence-initial imperatives ("Make sure ..." -> "Are Making sure ..."); 4 sentence-initial firings of this kind occur in the raw output (they are kept where the field passes the gate).

## Rules fired (top 25; counts over all fields of all 300 items, raw output)

| rule (Multi-VALUE feature function) | times fired | items |
|---|---|---|
| fronting_pobj | 1450 | 158 |
| remove_det_definite | 1289 | 277 |
| mass_noun_plurals | 1161 | 245 |
| null_prepositions | 1069 | 164 |
| present_perfect_for_past | 613 | 168 |
| zero_plural | 440 | 136 |
| progressives | 431 | 181 |
| demonstrative_for_definite_articles | 411 | 143 |
| definite_for_indefinite_articles | 383 | 173 |
| acomp_focusing_like | 254 | 142 |
| remove_det_indefinite | 234 | 135 |
| null_referential_pronouns | 205 | 125 |
| reduced_relative | 177 | 33 |
| definite_abstract | 160 | 103 |
| regularized_plurals | 158 | 93 |
| verbal_ing_suffix | 134 | 88 |
| indef_one | 134 | 69 |
| plural_to_singular_human | 110 | 65 |
| indefinite_for_definite_articles | 102 | 59 |
| double_obj_order | 101 | 34 |
| zero_plural_after_quantifier | 86 | 58 |
| shadow_pronouns | 85 | 67 |
| null_genitive | 62 | 54 |
| null_relcl | 61 | 47 |
| yall | 55 | 28 |

(81 distinct rules fired in total; full table in `inde_rule_counts.csv`; per-field rule spans in `rules_detail` of the condition file.)

## Lowest-similarity fields (reverted by the gate)

- `https://en.wikivoyage.org/wiki/First_and_business_class_flights#2` field `mc_answer2` cos=0.400
- `https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes#1` field `mc_answer2` cos=0.496
- `https://en.wikivoyage.org/wiki/Returning_home#2` field `mc_answer3` cos=0.501
- `https://en.wikinews.org/wiki/Militants_bomb_Gaziantep,_Turkey_police_headquarters#2` field `mc_answer1` cos=0.527
- `https://en.wikibooks.org/wiki/Wikijunior:Dinosaurs/Triceratops#1` field `mc_answer1` cos=0.578
- `https://en.wikinews.org/wiki/Scottish_Justice_Secretary_%27acutely_aware_of_unusual_publicity%27_in_Kular_case#1` field `question` cos=0.601
- `https://en.wikivoyage.org/wiki/Timbuktu#1` field `mc_answer2` cos=0.634
- `https://en.wikinews.org/wiki/Scientists_report_two_life-saving_treatments_for_Ebola#1` field `mc_answer1` cos=0.662
- `https://en.wikibooks.org/wiki/Communication_Theory/Uses_and_Gratifications#1` field `mc_answer1` cos=0.674
- `https://en.wikivoyage.org/wiki/Downhill_snowsports#1` field `mc_answer3` cos=0.689

## Three examples (original -> final file; raw output shown when the gate reverted a field)

### `https://en.wikinews.org/wiki/Sealing_ships_trapped_in_ice_off_coast_of_Newfoundland#2`  (rules fired: 25)

- **flores_passage** (cos 0.9823, edit 0.1007)
  - original: In addition to the crushing ice, extreme weather conditions have been hampering rescue efforts. Pittman suggested that conditions wouldn't improve until sometime next week. The amount and thickness of the pack ice, according to Pittman, is the worst it has been for sealers in the past 15 years.
  - Indian English: In additions to that crushing ices, extreme weathers conditions have been hampering rescue efforts. Pittman had suggested that conditions wouldn't improve until sometime next week. That amount and thickness of that pack ices, according to Pittman, the most worst is for sealers in the past 15 year.
- **question** (cos 0.9154, edit 0.1452)
  - original: In addition to the ice, what else is affecting rescue efforts?
  - Indian English: In additions to ices, what else affecting rescue efforts?
- **mc_answer1** (cos 0.7689, edit 0.4118)
  - original: Lack of supplies
  - Indian English: Lack of supplies
  - raw Multi-VALUE (reverted by gate): The Lack supplies
- **mc_answer2** (cos 0.9922, edit 0.0526)
  - original: Weather conditions
  - Indian English: Weathers conditions
- **mc_answer3** (cos 0.8885, edit 0.9167)
  - original: Lack of help
  - Indian English: Of help Lack
- **mc_answer4** (cos 0.7651, edit 0.2727)
  - original: The sealers
  - Indian English: The sealers
  - raw Multi-VALUE (reverted by gate): A sealers
- rules: {"mass_noun_plurals": 7, "demonstrative_for_definite_articles": 3, "present_perfect_for_past": 1, "drop_copula_be_NP": 1, "double_superlative": 1, "null_referential_pronouns": 1, "present_for_exp_perfect": 2, "zero_plural_after_quantifier": 1, "remove_det_definite": 1, "drop_aux_be_progressive": 1, "definite_abstract": 1, "null_prepositions": 1, "fronting_pobj": 3, "indefinite_for_definite_articles": 1}

### `https://en.wikinews.org/wiki/BBC_newsreader_sacked_after_conviction_for_wounding_teenager#1`  (rules fired: 34)

- **flores_passage** (cos 0.9901, edit 0.2222)
  - original: "The trial took place at Birmingham Crown Court and concluded on August 3. The presenter, who was arrested at the scene, denied the attack and claimed he used the pole to protect himself from bottles being thrown at him by up to thirty people. Blake was also convicted of attempting to pervert the course of justice. The judge told Blake it was ""almost inevitable"" he was going to be sent to jail."
  - Indian English: " trial had taken place at Birmingham Crown Court and have concluded on August 3. presenter, who have been arrested it at scene, had denied attack and have claimed had used pole to protect himself from bottles being thrown at him by up to thirty person. Blake have been also convicted of attempting to pervert course of the justice. judge had told Blake it had been ""like almost inevitable"" was going to be sent to jail."
- **question** (cos 0.9779, edit 0.1268)
  - original: According to the passage, which statement about Blake is not accurate?
  - Indian English: According to passage, which statement about Blake is not like accurate?
- **mc_answer1** (cos 0.9457, edit 0.4839)
  - original: He was arrested at the scene
  - Indian English: He have been arresting at scene
- **mc_answer2** (cos 0.8842, edit 0.375)
  - original: He admitted to the attack
  - Indian English: Himself had admitted to a attack
- **mc_answer3** (cos 0.9126, edit 0.4286)
  - original: He used a pole during the attack
  - Indian English: Himself had used pole during attack
- **mc_answer4** (cos 1.0, edit 0.0)
  - original: He claimed that bottles were thrown at him
  - Indian English: He claimed that bottles were thrown at him
- rules: {"remove_det_definite": 10, "present_perfect_for_past": 12, "shadow_pronouns": 1, "null_referential_pronouns": 2, "zero_plural_after_quantifier": 1, "definite_abstract": 1, "acomp_focusing_like": 2, "verbal_ing_suffix": 1, "absolute_reflex": 2, "indefinite_for_definite_articles": 1, "remove_det_indefinite": 1}

### `https://en.wikinews.org/wiki/Australian_Big_Brother_cleared_of_breaching_content_laws#2`  (rules fired: 45)

- **flores_passage** (cos 0.9695, edit 0.209)
  - original: The ACMA also found that despite the video being streamed on the Internet, Big Brother had not breached online content censorship laws as the media had not been stored on Big Brother's website. The Broadcasting Services Act provides for the regulation of Internet content, however to be considered Internet content, it must physically reside on a server.
  - Indian English: ACMA also had found that videos being streamed Internets, Big Brother had not breached online contents censorships laws as mediums had not been stored Big Brother website. Broadcasting Services Act is providing regulations Internets contents, however to be considered Internets contents, must physically reside the server.
- **question** (cos 0.9807, edit 0.2651)
  - original: What parameters must media fall under in order to be considered internet content?
  - Indian English: What parameters must mediums in order to be considered internet content fall under?
- **mc_answer1** (cos 0.9828, edit 0.1429)
  - original: It must be streamed on the internet
  - Indian English: It must be streamed on internets
- **mc_answer2** (cos 0.991, edit 0.0769)
  - original: It must physically reside on a server
  - Indian English: It must physically reside on the server
- **mc_answer3** (cos 0.9253, edit 0.2188)
  - original: It cannot be stored on a website
  - Indian English: It cannot be storing the website
- **mc_answer4** (cos 0.9914, edit 0.129)
  - original: It must be approved by the ACMA
  - Indian English: It must be approved by ACMA
- rules: {"remove_det_definite": 8, "present_perfect_for_past": 1, "null_prepositions": 7, "mass_noun_plurals": 10, "double_past": 2, "regularized_plurals": 2, "null_genitive": 1, "progressives": 1, "null_referential_pronouns": 1, "definite_for_indefinite_articles": 3, "fronting_pobj": 8, "verbal_ing_suffix": 1}
