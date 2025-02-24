import React, { useState } from "react";
import DropDown from "./DropDown";

const Menu: React.FC = (): JSX.Element => {
  const [showDropDown, setShowDropDown] = useState<boolean>(false);
  const [selectSample, setSelectSample] = useState<string>("");
  const samples = () => {
    return ["DLPFC_Br8667_mid_manual_alignment_all", "DLPFC_Br2720_ant_2", "DLPFC_Br2743_post_manual_alignment", "DLPFC_Br6471_ant_manual_alignment_all"];
  };

  /**
   * Toggle the drop down menu
   */
  const toggleDropDown = () => {
    setShowDropDown(!showDropDown);
  };

  /**
   * Hide the drop down menu if click occurs
   * outside of the drop-down element.
   *
   * @param event  The mouse event
   */
  const dismissHandler = (event: React.FocusEvent<HTMLButtonElement>): void => {
    if (event.currentTarget === event.target) {
      setShowDropDown(false);
    }
  };

  /**
   * Callback function to consume the
   * sample name from the child component
   *
   * @param sample  The selected sample
   */
  const sampleSelection = (sample: string): void => {
    setSelectSample(sample);
  };

  return (
    <>
      <div className="announcement">
        <div>
          {selectSample
            ? `You selected ${selectSample} to visualize`
            : "Select your visualization"}
        </div>
      </div>
      <button
        className={showDropDown ? "active" : undefined}
        onClick={(): void => toggleDropDown()}
        onBlur={(e: React.FocusEvent<HTMLButtonElement>): void =>
          dismissHandler(e)
        }
      >
        <div>{selectSample ? "Select: " + selectSample : "Select ..."} </div>
        {showDropDown && (
          <DropDown
            samples={samples()}
            showDropDown={false}
            toggleDropDown={(): void => toggleDropDown()}
            sampleSelection={sampleSelection}
          />
        )}
      </button>
    </>
  );
};

export default Menu;
