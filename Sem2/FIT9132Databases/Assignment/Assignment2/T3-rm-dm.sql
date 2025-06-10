--****PLEASE ENTER YOUR DETAILS BELOW****
--T3-rm-dm.sql

--Student ID:
--Student Name:

/* Comments for your marker:




*/

-- (a) Add a new competitor called Jackson Bull (3 marks)
-- Note: In a real system, you would use a database sequence for comp_no to ensure uniqueness in concurrent environments.
-- Using MAX(comp_no) + 1 is for simplicity as requested, but can lead to issues in multi-user scenarios.
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone)
VALUES (
    (SELECT NVL(MAX(comp_no), 0) + 1 FROM COMPETITOR), -- Generates next comp_no
    'Jackson',
    'Bull',
    'M',
    TO_DATE('15-MAY-2000', 'DD-MON-YYYY'),
    'jackson.bull@monash.edu',
    'Y',
    '0422412524'
);

-- Confirmation for (a)
SELECT * FROM COMPETITOR WHERE comp_phone = '0422412524';


-- (b) Create a new team 'The Great Runners' with Jackson as leader (3 marks)
-- This involves inserting into TEAM and then inserting Jackson's ENTRY record linking to the team.

-- Get common values to avoid repeated subqueries.
-- For simple SQL, these values are obtained via subqueries directly in the DML.
-- Carnival Date for RM Winter Series Caulfield 2025:
-- (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025') -> TO_DATE('29-JUN-2025', 'DD-MON-YYYY')
-- Event ID for 5 Km Run in that carnival:
-- (SELECT event_id FROM EVENT WHERE eventtype_code = '5K' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')) -> should return a single event_id, e.g., 11

-- Insert the new team 'The Great Runners'.
-- Ensure a unique team_id. Using MAX+1 for simplicity.
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no)
VALUES (
    (SELECT NVL(MAX(team_id), 0) + 1 FROM TEAM), -- New team_id
    'The Great Runners',
    (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'),
    (SELECT event_id FROM EVENT WHERE eventtype_code = '5K' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')),
    (SELECT NVL(MAX(entry_no), 0) + 1 FROM ENTRY WHERE event_id = (SELECT event_id FROM EVENT WHERE eventtype_code = '5K' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'))) -- Provisional entry_no for leader (used by TEAM.team_entry_fk)
);

-- Insert Jackson's entry as the team leader.
-- This entry_no should match the one provisionally used in the TEAM insert above.
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
VALUES (
    (SELECT event_id FROM EVENT WHERE eventtype_code = '5K' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')), -- Event ID for 5K
    (SELECT entry_no FROM TEAM WHERE team_name = 'The Great Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')), -- Entry_no used for team leader
    TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'),
    TO_TIMESTAMP('08:30:00', 'HH24:MI:SS'),
    INTERVAL '0 00:30:00' DAY TO SECOND,
    (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524'), -- Jackson's comp_no
    (SELECT team_id FROM TEAM WHERE team_name = 'The Great Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')), -- Team ID for The Great Runners
    (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue')
);

-- Confirmation for (b)
SELECT T.team_name, C.comp_fname, C.comp_lname, E.entry_no
FROM TEAM T
JOIN ENTRY E ON T.event_id = E.event_id AND T.entry_no = E.entry_no -- Join TEAM to leader's ENTRY
JOIN COMPETITOR C ON E.comp_no = C.comp_no
WHERE T.team_name = 'The Great Runners'
AND T.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025');


-- (c) Another competitor, Steve Bull, joins The Great Runners (4 marks)

-- First, add Steve Bull as a new competitor if he doesn't exist.
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone)
SELECT
    (SELECT NVL(MAX(comp_no), 0) + 1 FROM COMPETITOR),
    'Steve',
    'Bull',
    'M',
    TO_DATE('20-APR-1997', 'DD-MON-YYYY'),
    'steve.bull@example.com',
    'N',
    '0422251427'
FROM DUAL -- DUAL is a dummy table used for single-row queries
WHERE NOT EXISTS (SELECT 1 FROM COMPETITOR WHERE comp_phone = '0422251427');

-- Insert Steve's entry into the 10 Km Run event and link him to 'The Great Runners' team.
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
VALUES (
    (SELECT event_id FROM EVENT WHERE eventtype_code = '10K' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')), -- Event ID for 10K
    (SELECT NVL(MAX(entry_no), 0) + 1 FROM ENTRY WHERE event_id = (SELECT event_id FROM EVENT WHERE eventtype_code = '10K' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'))), -- New entry_no for Steve
    TO_TIMESTAMP('08:15:00', 'HH24:MI:SS'),
    TO_TIMESTAMP('09:20:00', 'HH24:MI:SS'),
    INTERVAL '0 01:05:00' DAY TO SECOND,
    (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422251427'), -- Steve's comp_no
    (SELECT team_id FROM TEAM WHERE team_name = 'The Great Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')), -- Team ID for The Great Runners
    (SELECT char_id FROM CHARITY WHERE char_name = 'Heart Foundation')
);

-- Confirmation for (c) - Show all members of 'The Great Runners'
SELECT C.comp_fname, C.comp_lname, T.team_name, E.event_id, E.entry_no
FROM ENTRY E
JOIN COMPETITOR C ON E.comp_no = C.comp_no
JOIN TEAM T ON E.team_id = T.team_id
WHERE T.team_name = 'The Great Runners'
AND T.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025');


-- (d) Keith Withdraws, Super Runners Disband (4 marks)

-- Step 1: Nullify team_id for ALL entries that belong to the 'Super Runners' team.
-- This ensures that the ENTRY_TEAM_FK constraint is not violated when deleting the TEAM.
UPDATE ENTRY
SET team_id = NULL
WHERE team_id = (SELECT team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'));

-- Step 2: Delete the TEAM record for 'Super Runners'.
-- This will now succeed because no ENTRY records are referencing it.
DELETE FROM TEAM
WHERE team_id = (SELECT team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'));

-- Step 3: Delete Keith's entry for the RM Winter Series Caulfield 2025 carnival.
-- This will now succeed because the TEAM record that referenced it (as the leader) is gone.
DELETE FROM ENTRY
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422141112')
  AND event_id IN (SELECT event_id FROM EVENT WHERE carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'));

-- Confirmation for (d) - Check if Keith's entry and Super Runners team are gone
SELECT COUNT(*) FROM ENTRY WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422141112') AND event_id IN (SELECT event_id FROM EVENT WHERE carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'));
SELECT COUNT(*) FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025');