--****PLEASE ENTER YOUR DETAILS BELOW****
--T3-rm-dm.sql

--Student ID:
--Student Name:

/* Comments for your marker:




*/

--(a) Create Sequences
-- Drop sequence for COMPETITOR if it exists
DROP SEQUENCE competitor_seq;

-- Create sequence for COMPETITOR
CREATE SEQUENCE competitor_seq
START WITH 100
INCREMENT BY 5;

-- Drop sequence for TEAM if it exists
DROP SEQUENCE team_seq;

-- Create sequence for TEAM
CREATE SEQUENCE team_seq
START WITH 100
INCREMENT BY 5;

-- (b) Record New Competitors, Team, and Entries (8 marks)
SET TRANSACTION NAME 'Register Keith and Jackson, form Super Runners';

DECLARE
    v_carn_date       DATE;
    v_10km_event_id   NUMBER;
    v_char_id_sa      NUMBER;
    v_char_id_rspca   NUMBER;
    v_keith_comp_no   NUMBER;
    v_jackson_comp_no NUMBER;
    v_keith_entry_no  NUMBER;
    v_jackson_entry_no NUMBER;
    v_team_id         NUMBER;
BEGIN
    -- Retrieve necessary IDs and values
    SELECT carn_date INTO v_carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025';

    SELECT e.event_id INTO v_10km_event_id
    FROM EVENT e
    JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
    WHERE e.carn_date = v_carn_date
      AND et.eventtype_desc = '10 Km Run';

    SELECT char_id INTO v_char_id_sa FROM CHARITY WHERE char_name = 'Salvation Army';
    SELECT char_id INTO v_char_id_rspca FROM CHARITY WHERE char_name = 'RSPCA';

    -- Insert Keith Rose
    INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone)
    VALUES (competitor_seq.NEXTVAL, 'Keith', 'Rose', 'M', TO_DATE('15-JAN-1990', 'DD-MON-YYYY'), 'keith.rose@monash.edu', 'Y', '0422141112')
    RETURNING comp_no INTO v_keith_comp_no;

    -- Insert Jackson Bull
    INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone)
    VALUES (competitor_seq.NEXTVAL, 'Jackson', 'Bull', 'M', TO_DATE('20-MAR-1992', 'DD-MON-YYYY'), 'jackson.bull@monash.edu', 'Y', '0422412524')
    RETURNING comp_no INTO v_jackson_comp_no;

    -- Determine next entry_no for Keith's 10 Km Run entry
    SELECT NVL(MAX(entry_no), 0) + 1 INTO v_keith_entry_no
    FROM ENTRY
    WHERE event_id = v_10km_event_id;

    -- Insert Keith's entry (as team leader)
    INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
    VALUES (v_10km_event_id, v_keith_entry_no, TO_TIMESTAMP('08:30:00', 'HH24:MI:SS'), NULL, NULL, v_keith_comp_no, NULL, v_char_id_sa);

    -- Insert team (leader's entry details are part of TEAM's FK)
    INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no)
    VALUES (team_seq.NEXTVAL, 'Super Runners', v_carn_date, v_10km_event_id, v_keith_entry_no)
    RETURNING team_id INTO v_team_id;

    -- Update Keith's entry to set team_id (must be done after team insert)
    UPDATE ENTRY
    SET team_id = v_team_id
    WHERE event_id = v_10km_event_id
      AND entry_no = v_keith_entry_no;

    -- Determine next entry_no for Jackson's 10 Km Run entry
    SELECT NVL(MAX(entry_no), 0) + 1 INTO v_jackson_entry_no
    FROM ENTRY
    WHERE event_id = v_10km_event_id;

    -- Insert Jackson's entry
    INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
    VALUES (v_10km_event_id, v_jackson_entry_no, TO_TIMESTAMP('08:30:05', 'HH24:MI:SS'), NULL, NULL, v_jackson_comp_no, v_team_id, v_char_id_rspca);

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE; -- Re-raise the exception to indicate failure
END;
/

-- (c) Update Jackson Bull's Entry (2 marks)
SET TRANSACTION NAME 'Update Jackson Bull''s entry';

DECLARE
    v_carn_date        DATE;
    v_old_10km_event_id NUMBER;
    v_new_5km_event_id NUMBER;
    v_char_id_beyond_blue NUMBER;
    v_jackson_comp_no   NUMBER;
    v_jackson_new_entry_no NUMBER;
BEGIN
    SELECT carn_date INTO v_carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025';

    SELECT e.event_id INTO v_old_10km_event_id
    FROM EVENT e
    JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
    WHERE e.carn_date = v_carn_date
      AND et.eventtype_desc = '10 Km Run';

    SELECT e.event_id INTO v_new_5km_event_id
    FROM EVENT e
    JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
    WHERE e.carn_date = v_carn_date
      AND et.eventtype_desc = '5 Km Run';

    SELECT char_id INTO v_char_id_beyond_blue FROM CHARITY WHERE char_name = 'Beyond Blue';
    SELECT comp_no INTO v_jackson_comp_no FROM COMPETITOR WHERE comp_phone = '0422412524';

    -- Delete Jackson's old entry for the 10 Km Run
    DELETE FROM ENTRY
    WHERE comp_no = v_jackson_comp_no
      AND event_id = v_old_10km_event_id;

    -- Determine next entry_no for Jackson's new 5K entry
    SELECT NVL(MAX(entry_no), 0) + 1 INTO v_jackson_new_entry_no
    FROM ENTRY
    WHERE event_id = v_new_5km_event_id;

    -- Insert Jackson's new entry for the 5 Km Run (as individual runner)
    INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
    VALUES (v_new_5km_event_id, v_jackson_new_entry_no, TO_TIMESTAMP('08:45:00', 'HH24:MI:SS'), NULL, NULL, v_jackson_comp_no, NULL, v_char_id_beyond_blue);

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE; -- Re-raise the exception
END;
/

-- (d) Keith Withdraws, Team Disbands (4 marks)
SET TRANSACTION NAME 'Keith withdraws, Super Runners disband';

DECLARE
    v_keith_comp_no   NUMBER;
    v_jackson_comp_no NUMBER;
    v_carn_date       DATE;
    v_super_runners_team_id NUMBER;
BEGIN
    SELECT comp_no INTO v_keith_comp_no FROM COMPETITOR WHERE comp_phone = '0422141112';
    SELECT comp_no INTO v_jackson_comp_no FROM COMPETITOR WHERE comp_phone = '0422412524';
    SELECT carn_date INTO v_carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025';
    SELECT team_id INTO v_super_runners_team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = v_carn_date;

    -- Update Jackson's entry to remove him from the team (if still assigned)
    UPDATE ENTRY
    SET team_id = NULL
    WHERE comp_no = v_jackson_comp_no
      AND team_id = v_super_runners_team_id; -- Target specific team if exists

    -- Delete Keith's entry for the RM Winter Series Caulfield 2025 carnival
    DELETE FROM ENTRY
    WHERE comp_no = v_keith_comp_no
      AND event_id IN (SELECT event_id FROM EVENT WHERE carn_date = v_carn_date);

    -- Delete the Super Runners team for the RM Winter Series Caulfield 2025 carnival
    DELETE FROM TEAM
    WHERE team_id = v_super_runners_team_id;

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE; -- Re-raise the exception
END;
/