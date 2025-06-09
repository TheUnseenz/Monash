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
-- Assuming:
--   'RM SUMMER SERIES CLAYTON 2024' (01-Jan-2024)
--   'RM AUTUMN SERIES CAULFIELD 2024' (01-Apr-2024)
--   'RM WINTER SERIES CAULFIELD 2025' (29-Jun-2025)
--   'RM SPRING SERIES CLAYTON 2024' (01-Oct-2024)
--   'RM AUTUMN SERIES CLAYTON 2025' (01-Apr-2025)
-- event_id and entry_no for team leader will be from ENTRY table below
-- -----------------------------------------------------------------------------

-- Team 1: Speedy Gazelles (RM SUMMER SERIES CLAYTON 2024) - leader: Alice Smith (comp_no 1), Entry in Event 1 (10 Km Run) entry_no 1
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no) VALUES (1, 'Speedy Gazelles', TO_DATE('01-JAN-2024', 'DD-MON-YYYY'), 1, 1);

-- Team 2: Roadrunners (RM AUTUMN SERIES CAULFIELD 2024) - leader: Bob Johnson (comp_no 2), Entry in Event 3 (10 Km Run) entry_no 1
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no) VALUES (2, 'Roadrunners', TO_DATE('01-APR-2024', 'DD-MON-YYYY'), 3, 1);

-- Team 3: Trail Blazers (RM WINTER SERIES CAULFIELD 2025) - leader: Charlie Brown (comp_no 3), Entry in Event 5 (10 Km Run) entry_no 1
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no) VALUES (3, 'Trail Blazers', TO_DATE('29-JUN-2025', 'DD-MON-YYYY'), 5, 1);

-- Team 4: Lone Wolves (RM SPRING SERIES CLAYTON 2024) - leader: Diana Prince (comp_no 4), Entry in Event 7 (Marathon) entry_no 1
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no) VALUES (4, 'Lone Wolves', TO_DATE('01-OCT-2024', 'DD-MON-YYYY'), 7, 1);

-- Team 5: Speedy Gazelles (RM AUTUMN SERIES CLAYTON 2025) - same team name, different carnival. leader: Eve Adams (comp_no 5), Entry in Event 8 (10 Km Run) entry_no 1
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no) VALUES (5, 'Speedy Gazelles', TO_DATE('01-APR-2025', 'DD-MON-YYYY'), 8, 1);


-- -----------------------------------------------------------------------------
-- ENTRY TABLE DATA
-- At least 30 entries
-- At least 10 different competitors (comp_no 1-10 used for diversity)
-- At least 6 events from 3 different carnivals (see event_id values below)
-- At least 5 competitors who join >2 events
-- At least 2 uncompleted entries (NULL finishtime/elapsedtime)
-- Use TO_TIMESTAMP for start/finish times, ensure sensible times
-- -----------------------------------------------------------------------------

-- Assuming Event IDs for test data:
-- Event 1: 10 Km Run, RM SUMMER SERIES CLAYTON 2024 (ID: 1001, Date: 01-JAN-2024)
-- Event 2: 5 Km Run, RM SUMMER SERIES CLAYTON 2024 (ID: 1002, Date: 01-JAN-2024)
-- Event 3: 10 Km Run, RM AUTUMN SERIES CAULFIELD 2024 (ID: 1003, Date: 01-APR-2024)
-- Event 4: 3 Km Community Run/Walk, RM AUTUMN SERIES CAULFIELD 2024 (ID: 1004, Date: 01-APR-2024)
-- Event 5: 10 Km Run, RM WINTER SERIES CAULFIELD 2025 (ID: 1005, Date: 29-JUN-2025) - Future
-- Event 6: 5 Km Run, RM WINTER SERIES CAULFIELD 2025 (ID: 1006, Date: 29-JUN-2025) - Future
-- Event 7: Marathon 42.2 Km, RM SPRING SERIES CLAYTON 2024 (ID: 1007, Date: 01-OCT-2024)
-- Event 8: 10 Km Run, RM AUTUMN SERIES CLAYTON 2025 (ID: 1008, Date: 01-APR-2025) - Future


-- Entries for Competitor 1 (Alice Smith) - Joins >2 events
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1001, 1, TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:55:30', 'HH24:MI:SS'), INTERVAL '0 00:55:30' DAY TO SECOND, 1, 1, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')); -- Speedy Gazelles leader
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1003, 2, TO_TIMESTAMP('08:05:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:00:15', 'HH24:MI:SS'), INTERVAL '0 00:55:15' DAY TO SECOND, 1, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')); -- Individual
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1007, 2, TO_TIMESTAMP('07:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('11:00:00', 'HH24:MI:SS'), INTERVAL '0 04:00:00' DAY TO SECOND, 1, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue'));

-- Entries for Competitor 2 (Bob Johnson) - Joins >2 events
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1003, 1, TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:58:00', 'HH24:MI:SS'), INTERVAL '0 00:58:00' DAY TO SECOND, 2, 2, (SELECT char_id FROM CHARITY WHERE char_name = 'Red Cross')); -- Roadrunners leader
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1001, 2, TO_TIMESTAMP('08:01:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:59:00', 'HH24:MI:SS'), INTERVAL '0 00:58:00' DAY TO SECOND, 2, 1, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF')); -- Speedy Gazelles member
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1008, 2, TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'), NULL, NULL, 2, 5, (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')); -- Uncompleted future entry (Team Speedy Gazelles)

-- Entries for Competitor 3 (Charlie Brown) - Joins >2 events
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1005, 1, TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'), NULL, NULL, 3, 3, (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue')); -- Trail Blazers leader (Uncompleted future entry)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1002, 1, TO_TIMESTAMP('09:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:25:00', 'HH24:MI:SS'), INTERVAL '0 00:25:00' DAY TO SECOND, 3, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army'));
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1004, 2, TO_TIMESTAMP('10:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('10:20:00', 'HH24:MI:SS'), INTERVAL '0 00:20:00' DAY TO SECOND, 3, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Red Cross'));

-- Entries for Competitor 4 (Diana Prince) - Joins >2 events
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1007, 1, TO_TIMESTAMP('07:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('11:30:00', 'HH24:MI:SS'), INTERVAL '0 04:30:00' DAY TO SECOND, 4, 4, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF')); -- Lone Wolves leader
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1001, 3, TO_TIMESTAMP('08:02:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:00:00', 'HH24:MI:SS'), INTERVAL '0 00:58:00' DAY TO SECOND, 4, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army'));
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1003, 3, TO_TIMESTAMP('08:06:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:05:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 4, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA'));

-- Entries for Competitor 5 (Eve Adams) - Joins >2 events
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1008, 1, TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:50:00', 'HH24:MI:SS'), INTERVAL '0 00:50:00' DAY TO SECOND, 5, 5, (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue')); -- Speedy Gazelles (2025) leader
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1002, 2, TO_TIMESTAMP('09:01:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:26:00', 'HH24:MI:SS'), INTERVAL '0 00:25:00' DAY TO SECOND, 5, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Red Cross'));
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1004, 3, TO_TIMESTAMP('10:01:00', 'HH24:MI:SS'), TO_TIMESTAMP('10:21:00', 'HH24:MI:SS'), INTERVAL '0 00:20:00' DAY TO SECOND, 5, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF'));

-- Other Competitor Entries to reach 30 total entries and use at least 10 different competitors
-- Comp 6 (Frank White)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1001, 4, TO_TIMESTAMP('08:03:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:02:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 6, 1, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')); -- Speedy Gazelles member
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1005, 2, TO_TIMESTAMP('08:05:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:05:00', 'HH24:MI:SS'), INTERVAL '0 01:00:00' DAY TO SECOND, 6, 3, (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue')); -- Trail Blazers member

-- Comp 7 (Grace Lee)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1003, 4, TO_TIMESTAMP('08:07:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:06:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 7, 2, (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')); -- Roadrunners member
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1006, 1, TO_TIMESTAMP('09:00:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:28:00', 'HH24:MI:SS'), INTERVAL '0 00:28:00' DAY TO SECOND, 7, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Red Cross'));

-- Comp 8 (Henry Green)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1007, 3, TO_TIMESTAMP('07:01:00', 'HH24:MI:SS'), TO_TIMESTAMP('11:45:00', 'HH24:MI:SS'), INTERVAL '0 04:44:00' DAY TO SECOND, 8, 4, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF')); -- Lone Wolves member
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1002, 3, TO_TIMESTAMP('09:02:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:27:00', 'HH24:MI:SS'), INTERVAL '0 00:25:00' DAY TO SECOND, 8, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army'));

-- Comp 9 (Ivy King)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1004, 4, TO_TIMESTAMP('10:02:00', 'HH24:MI:SS'), TO_TIMESTAMP('10:22:00', 'HH24:MI:SS'), INTERVAL '0 00:20:00' DAY TO SECOND, 9, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue'));
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1008, 3, TO_TIMESTAMP('08:01:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:52:00', 'HH24:MI:SS'), INTERVAL '0 00:51:00' DAY TO SECOND, 9, 5, (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')); -- Speedy Gazelles (2025) member

-- Comp 10 (Jack Wright)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1001, 5, TO_TIMESTAMP('08:04:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:03:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 10, 1, (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')); -- Speedy Gazelles member
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1003, 5, TO_TIMESTAMP('08:08:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:07:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 10, 2, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF')); -- Roadrunners member

-- Additional entries to reach 30 and ensure test coverage
-- Comp 11 (Karen Hall)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1005, 3, TO_TIMESTAMP('08:06:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:06:00', 'HH24:MI:SS'), INTERVAL '0 01:00:00' DAY TO SECOND, 11, 3, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')); -- Trail Blazers member
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1006, 2, TO_TIMESTAMP('09:01:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:29:00', 'HH24:MI:SS'), INTERVAL '0 00:28:00' DAY TO SECOND, 11, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Red Cross'));

-- Comp 12 (Liam Young)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1007, 4, TO_TIMESTAMP('07:02:00', 'HH24:MI:SS'), TO_TIMESTAMP('11:50:00', 'HH24:MI:SS'), INTERVAL '0 04:48:00' DAY TO SECOND, 12, 4, (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue')); -- Lone Wolves member
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1008, 4, TO_TIMESTAMP('08:02:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:53:00', 'HH24:MI:SS'), INTERVAL '0 00:51:00' DAY TO SECOND, 12, 5, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF')); -- Speedy Gazelles (2025) member

-- Comp 13 (Mia Scott)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1001, 6, TO_TIMESTAMP('08:05:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:04:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 13, 1, (SELECT char_id FROM CHARITY WHERE char_name = 'Red Cross')); -- Speedy Gazelles member
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1003, 6, TO_TIMESTAMP('08:09:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:08:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 13, 2, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')); -- Roadrunners member

-- Comp 14 (Noah Allen)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1005, 4, TO_TIMESTAMP('08:07:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:07:00', 'HH24:MI:SS'), INTERVAL '0 01:00:00' DAY TO SECOND, 14, 3, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF')); -- Trail Blazers member
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1006, 3, TO_TIMESTAMP('09:02:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:30:00', 'HH24:MI:SS'), INTERVAL '0 00:28:00' DAY TO SECOND, 14, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue'));

-- Comp 15 (Olivia Baker)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1007, 5, TO_TIMESTAMP('07:03:00', 'HH24:MI:SS'), TO_TIMESTAMP('11:55:00', 'HH24:MI:SS'), INTERVAL '0 04:52:00' DAY TO SECOND, 15, 4, (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')); -- Lone Wolves member
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1008, 5, TO_TIMESTAMP('08:03:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:54:00', 'HH24:MI:SS'), INTERVAL '0 00:51:00' DAY TO SECOND, 15, 5, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')); -- Speedy Gazelles (2025) member

-- More entries to reach 30+ total, ensuring various conditions
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1001, 7, TO_TIMESTAMP('08:06:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:05:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 1, 1, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF')); -- Comp 1 (Alice Smith) in another event/team
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1002, 4, TO_TIMESTAMP('09:03:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:28:00', 'HH24:MI:SS'), INTERVAL '0 00:25:00' DAY TO SECOND, 2, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')); -- Comp 2 (Bob Johnson) individual
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1003, 7, TO_TIMESTAMP('08:10:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:09:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 3, 2, (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue')); -- Comp 3 (Charlie Brown) in Roadrunners
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1004, 5, TO_TIMESTAMP('10:03:00', 'HH24:MI:SS'), TO_TIMESTAMP('10:23:00', 'HH24:MI:SS'), INTERVAL '0 00:20:00' DAY TO SECOND, 4, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Red Cross')); -- Comp 4 (Diana Prince) individual
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1005, 5, TO_TIMESTAMP('08:08:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:08:00', 'HH24:MI:SS'), INTERVAL '0 01:00:00' DAY TO SECOND, 5, 3, (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')); -- Comp 5 (Eve Adams) in Trail Blazers
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1006, 4, TO_TIMESTAMP('09:03:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:31:00', 'HH24:MI:SS'), INTERVAL '0 00:28:00' DAY TO SECOND, 6, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF')); -- Comp 6 (Frank White) individual
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1007, 6, TO_TIMESTAMP('07:04:00', 'HH24:MI:SS'), TO_TIMESTAMP('12:00:00', 'HH24:MI:SS'), INTERVAL '0 04:56:00' DAY TO SECOND, 7, 4, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')); -- Comp 7 (Grace Lee) in Lone Wolves
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1008, 6, TO_TIMESTAMP('08:04:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:55:00', 'HH24:MI:SS'), INTERVAL '0 00:51:00' DAY TO SECOND, 8, 5, (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue')); -- Comp 8 (Henry Green) in Speedy Gazelles (2025)

-- Additional entries to reach 30+
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1001, 8, TO_TIMESTAMP('08:07:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:06:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 9, 1, (SELECT char_id FROM CHARITY WHERE char_name = 'Red Cross')); -- Comp 9 in team
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1002, 5, TO_TIMESTAMP('09:04:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:29:00', 'HH24:MI:SS'), INTERVAL '0 00:25:00' DAY TO SECOND, 10, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF')); -- Comp 10 individual
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1003, 8, TO_TIMESTAMP('08:11:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:10:00', 'HH24:MI:SS'), INTERVAL '0 00:59:00' DAY TO SECOND, 11, 2, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')); -- Comp 11 in team
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1004, 6, TO_TIMESTAMP('10:04:00', 'HH24:MI:SS'), TO_TIMESTAMP('10:24:00', 'HH24:MI:SS'), INTERVAL '0 00:20:00' DAY TO SECOND, 12, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')); -- Comp 12 individual
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1005, 6, TO_TIMESTAMP('08:09:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:09:00', 'HH24:MI:SS'), INTERVAL '0 01:00:00' DAY TO SECOND, 13, 3, (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue')); -- Comp 13 in team
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1006, 5, TO_TIMESTAMP('09:04:00', 'HH24:MI:SS'), TO_TIMESTAMP('09:32:00', 'HH24:MI:SS'), INTERVAL '0 00:28:00' DAY TO SECOND, 14, NULL, (SELECT char_id FROM CHARITY WHERE char_name = 'Red Cross')); -- Comp 14 individual
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1007, 7, TO_TIMESTAMP('07:05:00', 'HH24:MI:SS'), TO_TIMESTAMP('12:05:00', 'HH24:MI:SS'), INTERVAL '0 05:00:00' DAY TO SECOND, 15, 4, (SELECT char_id FROM CHARITY WHERE char_name = 'WWF')); -- Comp 15 in team
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id) VALUES (1008, 7, TO_TIMESTAMP('08:05:00', 'HH24:MI:SS'), TO_TIMESTAMP('08:56:00', 'HH24:MI:SS'), INTERVAL '0 00:51:00' DAY TO SECOND, 1, 5, (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')); -- Comp 1 in team

COMMIT;
