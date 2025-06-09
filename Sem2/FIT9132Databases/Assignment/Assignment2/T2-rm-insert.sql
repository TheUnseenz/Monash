/*****PLEASE ENTER YOUR DETAILS BELOW*****/
--T2-rm-insert.sql

--Student ID:
--Student Name:

/* Comments for your marker:




*/

-- Task 2 Load the COMPETITOR, ENTRY and TEAM tables with your own
-- test data following the data requirements expressed in the brief

-- =======================================
-- COMPETITOR
-- =======================================



-- =======================================
-- ENTRY
-- =======================================



-- =======================================
-- TEAM
-- =======================================


SET TRANSACTION READ WRITE;

-- Task 2: Populate Sample Data

-- -----------------------------------------------------------------------------
-- COMPETITOR TABLE DATA
-- At least 15 competitors: 5 Monash, 5 Non-Monash, 5 additional
-- comp_no hardcoded below 100
-- -----------------------------------------------------------------------------

INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (1, 'Alice', 'Smith', 'F', TO_DATE('1998-03-15', 'YYYY-MM-DD'), 'alice.smith@monash.edu', 'Y', '0400111222');
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (2, 'Bob', 'Johnson', 'M', TO_DATE('1995-07-22', 'YYYY-MM-DD'), 'bob.johnson@monash.edu', 'Y', '0400333444');
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (3, 'Charlie', 'Brown', 'M', TO_DATE('2000-01-01', 'YYYY-MM-DD'), 'charlie.brown@monash.edu', 'Y', '0400555666');
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (4, 'Diana', 'Prince', 'F', TO_DATE('1997-11-11', 'YYYY-MM-DD'), 'diana.prince@monash.edu', 'Y', '0400777888');
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (5, 'Eve', 'Adams', 'F', TO_DATE('1996-09-01', 'YYYY-MM-DD'), 'eve.adams@monash.edu', 'Y', '0400999000');

INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (6, 'Frank', 'White', 'M', TO_DATE('1985-04-20', 'YYYY-MM-DD'), 'frank.white@example.com', 'N', '0410111222');
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (7, 'Grace', 'Lee', 'F', TO_DATE('1990-12-05', 'YYYY-MM-DD'), 'grace.lee@example.com', 'N', '0410333444');
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (8, 'Henry', 'Green', 'M', TO_DATE('1978-02-28', 'YYYY-MM-DD'), 'henry.green@example.com', 'N', '0410555666');
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (9, 'Ivy', 'King', 'F', TO_DATE('2001-06-10', 'YYYY-MM-DD'), 'ivy.king@example.com', 'N', '0410777888');
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (10, 'Jack', 'Wright', 'M', TO_DATE('1993-08-08', 'YYYY-MM-DD'), 'jack.wright@example.com', 'N', '0410999000');

INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (11, 'Karen', 'Hall', 'F', TO_DATE('1989-01-25', 'YYYY-MM-DD'), 'karen.hall@domain.com', 'N', '0420111222');
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (12, 'Liam', 'Young', 'M', TO_DATE('1991-03-30', 'YYYY-MM-DD'), 'liam.young@domain.com', 'Y', '0420333444'); -- Additional Monash student
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (13, 'Mia', 'Scott', 'F', TO_DATE('1999-05-18', 'YYYY-MM-DD'), 'mia.scott@domain.com', 'N', '0420555666');
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (14, 'Noah', 'Allen', 'M', TO_DATE('1987-10-12', 'YYYY-MM-DD'), 'noah.allen@domain.com', 'Y', '0420777888'); -- Additional Monash student
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone) VALUES (15, 'Olivia', 'Baker', 'F', TO_DATE('1994-07-07', 'YYYY-MM-DD'), 'olivia.baker@domain.com', 'N', '0420999000');


-- -----------------------------------------------------------------------------
-- TEAM TABLE DATA
-- At least 5 teams, 2 with >2 members, 1 team name used in 2 different carnivals
-- team_id hardcoded below 100
-- Reference actual carn_date and event_id from rm-schema-insert.sql
-- -----------------------------------------------------------------------------

-- Team 1: Speedy Gazelles (RM Spring Series Clayton 2024) - leader: Alice Smith (comp_no 1), Entry in Event 2 (10 Km Run) entry_no 1
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no) VALUES (1, 'Speedy Gazelles', TO_DATE('22-SEP-2024', 'DD-MON-YYYY'), 2, 1);

-- Team 2: Roadrunners (RM Spring Series Caulfield 2024) - leader: Bob Johnson (comp_no 2), Entry in Event 4 (10 Km Run) entry_no 1
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no) VALUES (2, 'Roadrunners', TO_DATE('05-OCT-2024', 'DD-MON-YYYY'), 4, 1);

-- Team 3: Trail Blazers (RM Winter Series Caulfield 2025) - leader: Charlie Brown (comp_no 3), Entry in Event 13 (10 Km Run) entry_no 1
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no) VALUES (3, 'Trail Blazers', TO_DATE('29-JUN-2025', 'DD-MON-YYYY'), 13, 1);

-- Team 4: Lone Wolves (RM Autumn Series Clayton 2025) - leader: Diana Prince (comp_no 4), Entry in Event 11 (Marathon) entry_no 1
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no) VALUES (4, 'Lone Wolves', TO_DATE('15-MAR-2025', 'DD-MON-YYYY'), 11, 1);

-- Team 5: Speedy Gazelles (RM Summer Series Caulfield 2025) - same team name, different carnival. leader: Eve Adams (comp_no 5), Entry in Event 8 (10 Km Run) entry_no 1
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no) VALUES (5, 'Speedy Gazelles', TO_DATE('02-FEB-2025', 'DD-MON-YYYY'), 8, 1);


-- -----------------------------------------------------------------------------
-- ENTRY TABLE DATA
-- At least 30 entries
-- At least 10 different competitors (comp_no 1-10 used for diversity)
-- At least 6 events from 3 different carnivals
-- At least 5 competitors who join >2 events
-- At least 2 uncompleted entries (NULL finishtime/elapsedtime)
-- Use TO_TIMESTAMP for start/finish times, ensure sensible times
-- Reference actual event_id and char_id from rm-schema-insert.sql
-- -----------------------------------------------------------------------------

-- Events from rm-schema-insert.sql and their carnivals:
-- Event 1: 5K, RM Spring Series Clayton 2024 (22-SEP-2024)
-- Event 2: 10K, RM Spring Series Clayton 2024 (22-SEP-2024)
-- Event 3: 5K, RM Spring Series Caulfield 2024 (05-OCT-2024)
-- Event 4: 10K, RM Spring Series Caulfield 2024 (05-OCT-2024)
-- Event 5: 21K, RM Spring Series Caulfield 2024 (05-OCT-2024)
-- Event 6: 3K, RM Summer Series Caulfield 2025 (02-FEB-2025)
-- Event 7: 5K, RM Summer Series Caulfield 2025 (02-FEB-2025)
-- Event 8: 10K, RM Summer Series Caulfield 2025 (02-FEB-2025)
-- Event 9: 21K, RM Summer Series Caulfield 2025 (02-FEB-2025)
-- Event 10: 3K, RM Autumn Series Clayton 2025 (15-MAR-2025)
-- Event 11: 42K, RM Autumn Series Clayton 2025 (15-MAR-2025)
-- Event 12: 5K, RM Winter Series Caulfield 2025 (29-JUN-2025)
-- Event 13: 10K, RM Winter Series Caulfield 2025 (29-JUN-2025)
-- Event 14: 21K, RM Winter Series Caulfield 2025 (29-JUN-2025)

-- Charities from rm-schema-insert.sql:
-- RSPCA (ID: 1)
-- Beyond Blue (ID: 2)
-- Salvation Army (ID: 3)
-- Amnesty International (ID: 4)


-- Entries for Competitor 1 (Alice Smith) - Joins >2 events
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (2, 1, TO_TIMESTAMP('08:30:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:25:30', 'HH24:MI:SS'), INTERVAL '0 00:55:30' DAY TO SECOND, 1, 1, 3); -- Speedy Gazelles leader, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (4, 2, TO_TIMESTAMP('08:35:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:30:15', 'HH24:MI:SS'), INTERVAL '0 00:55:15' DAY TO SECOND, 1, NULL, 1); -- Individual, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (11, 2, TO_TIMESTAMP('07:45:00', 'HH24:MI:SS'), TO_TIMESTAMP('11:45:00', 'HH24:MI:SS'), INTERVAL '0 04:00:00' DAY TO SECOND, 1, NULL, 2); -- Marathon

-- Entries for Competitor 2 (Bob Johnson) - Joins >2 events
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (4, 1, TO_TIMESTAMP('08:30:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:28:00', 'HH24:MI:SS'), INTERVAL '0 00:58:00' DAY TO SECOND, 2, 2, 4); -- Roadrunners leader, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (2, 2, TO_TIMESTAMP('08:31:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:29:00', 'HH24:MI:SS'), INTERVAL '0 00:58:00' DAY TO SECOND, 2, 1, 1); -- Speedy Gazelles member, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (8, 2, TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'), NULL, NULL, 2, 5, 1); -- Uncompleted future entry (Team Speedy Gazelles), 10K

-- Entries for Competitor 3 (Charlie Brown) - Joins >2 events
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (13, 1, TO_TIMESTAMP('08:30:00', 'HH24:MI:SS'), NULL, NULL, 3, 3, 2); -- Trail Blazers leader (Uncompleted future entry), 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1, 1, TO_TIMESTAMP('09:30:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:55:00', 'HH24:MI:SS'), INTERVAL '0 00:25:00' DAY TO SECOND, 3, NULL, 3); -- 5K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (6, 2, TO_TIMESTAMP('08:30:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:50:00', 'HH24:MI:SS'), INTERVAL '0 00:20:00' DAY TO SECOND, 3, NULL, 4); -- 3K

-- Entries for Competitor 4 (Diana Prince) - Joins >2 events
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (11, 1, TO_TIMESTAMP('07:45:00', 'HH24:MI:SS'), TO_TIMESTAMP('12:15:00', 'HH24:MI:SS'), INTERVAL '0 04:30:00' DAY TO SECOND, 4, 4, 1); -- Lone Wolves leader, Marathon
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (2, 3, TO_TIMESTAMP('08:32:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:30:00', 'HH24:MI:SS'), INTERVAL '0 00:58:00' DAY TO SECOND, 4, NULL, 3); -- 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (4, 3, TO_TIMESTAMP('08:36:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:35:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 4, NULL, 2); -- 10K

-- Entries for Competitor 5 (Eve Adams) - Joins >2 events
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (8, 1, TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:50:00', 'HH24:MI:SS'), INTERVAL '0 00:50:00' DAY TO SECOND, 5, 5, 2); -- Speedy Gazelles (2025) leader, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (3, 2, TO_TIMESTAMP('09:01:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:26:00', 'HH24:MI:SS'), INTERVAL '0 00:25:00' DAY TO SECOND, 5, NULL, 4); -- 5K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (7, 3, TO_TIMESTAMP('08:31:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:51:00', 'HH24:MI:SS'), INTERVAL '0 00:20:00' DAY TO SECOND, 5, NULL, 1); -- 5K

-- Other Competitor Entries to reach 30 total entries and use at least 10 different competitors
-- Comp 6 (Frank White)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (2, 4, TO_TIMESTAMP('08:33:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:32:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 6, 1, 3); -- Speedy Gazelles member, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (13, 2, TO_TIMESTAMP('08:35:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:35:00', 'HH24:MI:SS'), INTERVAL '0 01:00:00' DAY TO SECOND, 6, 3, 2); -- Trail Blazers member, 10K

-- Comp 7 (Grace Lee)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (4, 4, TO_TIMESTAMP('08:37:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:36:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 7, 2, 1); -- Roadrunners member, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (12, 1, TO_TIMESTAMP('08:45:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:13:00', 'HH24:MI:SS'), INTERVAL '0 00:28:00' DAY TO SECOND, 7, NULL, 4); -- 5K

-- Comp 8 (Henry Green)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (11, 3, TO_TIMESTAMP('07:46:00', 'HH24:MI:SS'), TO_TIMESTAMP('12:30:00', 'HH24:MI:SS'), INTERVAL '0 04:44:00' DAY TO SECOND, 8, 4, 1); -- Lone Wolves member, Marathon
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (3, 3, TO_TIMESTAMP('09:02:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:27:00', 'HH24:MI:SS'), INTERVAL '0 00:25:00' DAY TO SECOND, 8, NULL, 3); -- 5K

-- Comp 9 (Ivy King)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (6, 4, TO_TIMESTAMP('08:32:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:52:00', 'HH24:MI:SS'), INTERVAL '0 00:20:00' DAY TO SECOND, 9, NULL, 2); -- 3K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (8, 3, TO_TIMESTAMP('08:01:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:52:00', 'HH24:MI:SS'), INTERVAL '0 00:51:00' DAY TO SECOND, 9, 5, 1); -- Speedy Gazelles (2025) member, 10K

-- Comp 10 (Jack Wright)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (2, 5, TO_TIMESTAMP('08:34:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:33:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 10, 1, 1); -- Speedy Gazelles member, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (4, 5, TO_TIMESTAMP('08:38:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:37:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 10, 2, 4); -- Roadrunners member, 10K

-- Additional entries to reach 30 and ensure test coverage
-- Comp 11 (Karen Hall)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (13, 3, TO_TIMESTAMP('08:36:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:36:00', 'HH24:MI:SS'), INTERVAL '0 01:00:00' DAY TO SECOND, 11, 3, 3); -- Trail Blazers member, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (12, 2, TO_TIMESTAMP('08:46:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:14:00', 'HH24:MI:SS'), INTERVAL '0 00:28:00' DAY TO SECOND, 11, NULL, 4); -- 5K

-- Comp 12 (Liam Young)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (11, 4, TO_TIMESTAMP('07:47:00', 'HH24:MI:SS'), TO_TIMESTAMP('12:35:00', 'HH24:MI:SS'), INTERVAL '0 04:48:00' DAY TO SECOND, 12, 4, 2); -- Lone Wolves member, Marathon
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (8, 4, TO_TIMESTAMP('08:02:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:53:00', 'HH24:MI:SS'), INTERVAL '0 00:51:00' DAY TO SECOND, 12, 5, 1); -- Speedy Gazelles (2025) member, 10K

-- Comp 13 (Mia Scott)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (2, 6, TO_TIMESTAMP('08:35:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:34:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 13, 1, 4); -- Speedy Gazelles member, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (4, 6, TO_TIMESTAMP('08:39:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:38:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 13, 2, 3); -- Roadrunners member, 10K

-- Comp 14 (Noah Allen)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (13, 4, TO_TIMESTAMP('08:37:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:37:00', 'HH24:MI:SS'), INTERVAL '0 01:00:00' DAY TO SECOND, 14, 3, 1); -- Trail Blazers member, 10K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (12, 3, TO_TIMESTAMP('08:47:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:15:00', 'HH24:MI:SS'), INTERVAL '0 00:28:00' DAY TO SECOND, 14, NULL, 2); -- 5K

-- Comp 15 (Olivia Baker)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (11, 5, TO_TIMESTAMP('07:48:00', 'HH24:MI:SS'), TO_TIMESTAMP('12:40:00', 'HH24:MI:SS'), INTERVAL '0 04:52:00' DAY TO SECOND, 15, 4, 1); -- Lone Wolves member, Marathon
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (8, 5, TO_TIMESTAMP('08:03:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:54:00', 'HH24:MI:SS'), INTERVAL '0 00:51:00' DAY TO SECOND, 15, 5, 3); -- Speedy Gazelles (2025) member, 10K

-- More entries to reach 30+ total, ensuring various conditions
-- Comp 1 (Alice Smith) in another event/team
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (5, 1, TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:59:00', 'HH24:MI:SS'), INTERVAL '0 01:59:00' DAY TO SECOND, 1, NULL, 4); -- 21K
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (14, 1, TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('10:00:00', 'HH24:MI:SS'), INTERVAL '0 02:00:00' DAY TO SECOND, 1, NULL, 3); -- 21K

-- Comp 2 (Bob Johnson) individual
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1, 2, TO_TIMESTAMP('09:33:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:58:00', 'HH24:MI:SS'), INTERVAL '0 00:25:00' DAY TO SECOND, 2, NULL, 3); -- 5K

-- Comp 3 (Charlie Brown) in Roadrunners
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (4, 7, TO_TIMESTAMP('08:40:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:39:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 3, 2, 2); -- 10K

-- Comp 4 (Diana Prince) individual
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (6, 5, TO_TIMESTAMP('08:33:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:53:00', 'HH24:MI:SS'), INTERVAL '0 00:20:00' DAY TO SECOND, 4, NULL, 4); -- 3K

-- Comp 5 (Eve Adams) in Trail Blazers
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (13, 5, TO_TIMESTAMP('08:38:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:38:00', 'HH24:MI:SS'), INTERVAL '0 01:00:00' DAY TO SECOND, 5, 3, 1); -- 10K

-- Comp 6 (Frank White) individual
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (12, 4, TO_TIMESTAMP('08:48:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:16:00', 'HH24:MI:SS'), INTERVAL '0 00:28:00' DAY TO SECOND, 6, NULL, 4); -- 5K

-- Comp 7 (Grace Lee) in Lone Wolves
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (11, 6, TO_TIMESTAMP('07:49:00', 'HH24:MI:SS'), TO_TIMESTAMP('12:45:00', 'HH24:MI:SS'), INTERVAL '0 04:56:00' DAY TO SECOND, 7, 4, 3); -- Marathon

-- Comp 8 (Henry Green) in Speedy Gazelles (2025)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (8, 6, TO_TIMESTAMP('08:04:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:55:00', 'HH24:MI:SS'), INTERVAL '0 00:51:00' DAY TO SECOND, 8, 5, 2); -- 10K


COMMIT;
